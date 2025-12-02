"""
yolo_deepsort_facenet_opt.py

YOLOv8 + DeepSORT (or centroid fallback) + Stationary check + Camera-obstruct +
Face ID (face_recognition fallback -> facenet-pytorch) + persistence + suspicious capture +
timestamped snapshots + live preview + FPS sync + wide-object filter

To run :=> python yolo_deepsort_facenet_opt.py "D:\FaceTrackingProject\known face stationary check.mp4"

Optimizations added (toggleable):
 - RESIZE_WH: resize frames for inference and map coords back
 - DETECT_EVERY_N: run YOLO every N frames (tracker keeps IDs in between)
 - FACENET_EVERY_N: compute face embeddings only every N frames per track (async)
 - USE_INT8: try to load an int8/quantized model if provided
 - THREAD_POOL: compute embeddings & save snapshots in ThreadPoolExecutor
"""

import os
import cv2
import numpy as np
import sys
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, Future

# ---------------- CLI parsing (same as your previous) ----------------
_cmd_video_in = None
if len(sys.argv) > 1:
    _arg = sys.argv[1]
    try:
        _cmd_video_in = int(_arg)
    except Exception:
        _cmd_video_in = _arg
# --------------------------------------------------------------------

# ----- Optimization toggles & params (change these to experiment) -----
USE_INT8 = False                # Try to load a quantized yolov8n-int8.pt if True (falls back to yolov8n.pt)
RESIZE_WH = None   # changed from 640×360 to 960×540 to 1280x720 to 1600x900 to 1920x1080
# (width, height) for inference. Set to None to disable resizing.
DETECT_EVERY_N =  2 #2             # run object detection every N frames (1 = every frame)
FACENET_EVERY_N = 3 #3            # compute facenet embedding for a given track every N frames
THREAD_WORKERS = 2              # executor workers for async tasks (embedding & IO)
SKIP_FRAMES_FOR_WRITING = False # keep as False (we still write all frames). rarely change.

# ✅ NEW OPTIMIZATION PARAMETERS
DISABLE_PREVIEW = True          # If True, disables cv2.imshow for faster processing (toggleable)
STACK_LIMIT = 3                 # Max stack count before person becomes permanently suspicious
STACK_SECONDS_TO_INCREMENT = 2  # seconds that stationary must persist to auto-increment a stack
# ---------------------------------------------------------------------
# NEW: Read preview toggle from CLI
if "--preview" in sys.argv:
    DISABLE_PREVIEW = False
elif "--no-preview" in sys.argv:
    DISABLE_PREVIEW = True

# Try deep sort
use_deepsort = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    use_deepsort = True
    print("Using deep_sort_realtime.")
except Exception as e:
    print("deep_sort_realtime not available — will use centroid fallback. Err:", e)
    use_deepsort = False

# Try face_recognition
try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
    print("face_recognition available — will try using it first.")
except Exception as e:
    face_recognition = None
    FACE_RECOG_AVAILABLE = False
    print("face_recognition not imported:", e)

# facenet-pytorch fallback
FACENET_AVAILABLE = False
try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(keep_all=False, device=device)
    FACENET_AVAILABLE = True
    print("✅ facenet-pytorch ready on", device)
except Exception as e:
    mtcnn = None
    resnet = None
    print("facenet-pytorch not available:", e)

# ---------------- PARAMETERS (kept same as yours) ----------------
VIDEO_IN = _cmd_video_in if _cmd_video_in is not None else 0
VIDEO_OUT = "full_system_facenet_optimized.mp4"
MIN_CONF = 0.24 #from 35 to 25 to 27 to 25 to 20 to 27 to 26 to 24
STATIONARY_FRAMES = 60 # from 60 to 120 to 60
DISPLACEMENT_THRESHOLD = 10 # from 10 to 15 to 10
TARGET_CLASSES = [0]  # person
IGNORE_CLASSES = [39, 41, 67]
FR_EUCLIDEAN_THRESHOLD = 0.46 #47 to 46
FACENET_COSINE_THRESHOLD = 0.20
IOU_FILTER_THRESHOLD = 0.6
MAX_ASPECT_RATIO = 2.5 #from 1.2 to 2.5
# ---------------- Behavior persistence & snapshot ----------------
KNOWN_PERSIST_CONSEC = 8 # changed from 15 to 5 to 8 
KNOWN_PERSIST_NONCONSEC = 16 # changed from 30 to 10 to 16
SUSPICIOUS_SAVE_FOLDER = "suspicious_person"
SHOW_BOX_LAST_FRAMES = 1
# -----------------------------------------------------------------

os.makedirs(SUSPICIOUS_SAVE_FOLDER, exist_ok=True)

# ✅ NEW DATA STRUCTURES for stacks and permanent suspicious
suspicious_stacks = defaultdict(int)       # tid -> stack count (0..STACK_LIMIT)
stack_last_inc_frame = {}                  # tid -> last frame when stack incremented
permanent_suspicious = set()               # tids permanently suspicious (no more heavy processing)
# -----------------------------------------------------------------

# ------------------- load YOLO model (with int8 try) -------------------
# ✅ Added OpenVINO auto-load integration here (nothing else changed)
try:
    if os.path.exists("yolov8n_openvino_model"):
        print("Trying to load OpenVINO model: yolov8n_openvino_model")
        model = YOLO("yolov8n_openvino_model")
        print("✅ Using OpenVINO-optimized YOLOv8n model.")
    else:
        raise FileNotFoundError
except Exception as e:
    print("⚠️ OpenVINO model not found or failed to load, falling back:", e)
    model_file_candidates = []
    if USE_INT8:
        model_file_candidates.append("yolov8n-int8.pt")
    model_file_candidates.append("yolov8n.pt")

    loaded_model = None
    model_load_err = None
    for mf in model_file_candidates:
        try:
            print(f"Trying to load YOLO model: {mf}")
            loaded_model = YOLO(mf)
            print(f"Loaded model: {mf}")
            break
        except Exception as e:
            model_load_err = e
            print(f"Failed to load {mf}: {e}")
    if loaded_model is None:
        print("Falling back to YOLO('yolov8n.pt') via hub")
        model = YOLO("yolov8n.pt")
    else:
        model = loaded_model
# ---------------------------------------------------------------------

tracker = None
if use_deepsort:
    try:
        tracker = DeepSort(max_age=30)
    except Exception as e:
        print("Failed to init DeepSort:", e)
        tracker = None

centroid_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
last_seen_frame = {}
stationary_flag = {}
known_persistence = defaultdict(lambda: {"consec": 0, "total": 0})
known_persistent_ids = set()
recent_boxes = defaultdict(lambda: deque(maxlen=SHOW_BOX_LAST_FRAMES))
saved_suspicious = set()

# face db
known_face_entries = []  # (enc, name, 'fr'/'facenet')

def bbox_centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def normalize_vec(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def face_distance_euclidean(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

def face_distance_cosine(a, b):
    a = normalize_vec(a)
    b = normalize_vec(b)
    return 1.0 - float(np.dot(a, b))

# Async executor for embeddings and IO
executor = ThreadPoolExecutor(max_workers=max(1, THREAD_WORKERS))

# caches / bookkeeping for async embeddings
last_embedding = {}            # tid -> numpy vector
last_embedding_frame = {}      # tid -> frame_idx when embedding computed
embedding_future = {}          # tid -> Future (if ongoing)
embedding_lock = {}            # tid -> simple boolean guard (we'll just check presence)

def encode_with_facenet(img_rgb):
    """Synchronous embedding call (same as before). Input rgb uint8."""
    from PIL import Image
    import torchvision.transforms as T
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    pil = Image.fromarray(cv2.resize(img_rgb, (160, 160)))
    transform = T.Compose([
        T.Resize((160,160)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    x = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(x).cpu().numpy()[0]
    return emb.astype(np.float32)

# --- NEW: schedule_embedding_for_tid (re-inserted so file is complete) ---
def schedule_embedding_for_tid(tid, rgb_crop, current_frame):
    """
    Submits an embedding task for tid if none is running and it's due.
    Will update last_embedding and last_embedding_frame on completion.
    """
    # only submit if no existing future and no very recent embedding
    fut = embedding_future.get(tid)
    last_fr = last_embedding_frame.get(tid, -9999)
    if fut is None or fut.done():
        # submit only if it's been >= FACENET_EVERY_N frames since last embedding
        if (current_frame - last_fr) >= FACENET_EVERY_N:
            # submit the synchronous embedding function to executor
            future = executor.submit(encode_with_facenet, rgb_crop)
            embedding_future[tid] = future
            # attach callback to set caches when done (done callback runs in thread)
            def _on_done(f, tid=tid, frame=current_frame):
                try:
                    emb = f.result()
                    last_embedding[tid] = emb.astype(np.float32)
                    last_embedding_frame[tid] = frame
                except Exception as e:
                    # ignore embedding failures
                    print("Embedding failed for tid", tid, "->", e)
            future.add_done_callback(_on_done)
# -------------------------------------------------------------------------

def load_face_db(folder="faces_db"):
    if not os.path.exists(folder):
        print("No face DB folder:", folder)
        return
    for person in sorted(os.listdir(folder)):
        pdir = os.path.join(folder, person)
        if not os.path.isdir(pdir): continue
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
            path = os.path.join(pdir, fname)
            bgr = cv2.imread(path)
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            loaded = False
            if FACE_RECOG_AVAILABLE:
                try:
                    encs = face_recognition.face_encodings(rgb)
                    if len(encs) > 0:
                        known_face_entries.append((np.asarray(encs[0], dtype=np.float32), person, 'fr'))
                        loaded = True
                        print(f"Loaded FR encoding for {person} from {fname}")
                except Exception:
                    pass
            if not loaded and FACENET_AVAILABLE:
                try:
                    emb = encode_with_facenet(rgb)
                    known_face_entries.append((emb.astype(np.float32), person, 'facenet'))
                    print(f"Loaded facenet encoding for {person} from {fname}")
                except Exception:
                    pass
    print("✅ Loaded face encodings for", len(set([n for _, n, _ in known_face_entries])), "people.")

load_face_db("faces_db")

# ---------------- helpers ----------------
def camera_obstructed(frame, brightness_thresh=15, variance_thresh=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    var_val = float(np.var(gray))
    return (mean_val < brightness_thresh or mean_val > 255 - brightness_thresh or var_val < variance_thresh)

def iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(area1 + area2 - inter + 1e-6)

# --- NEW: async save helper (re-inserted) ---
def save_suspicious_snapshot_async(crop, tid):
    """Schedule saving suspicious crop to disk (non-blocking)."""
    def _save(img, tid_local):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(SUSPICIOUS_SAVE_FOLDER, f"suspicious_{tid_local}_{timestamp}.jpg")
            cv2.imwrite(snap_path, img)
            print(f"📸 Saved suspicious snapshot {snap_path}")
        except Exception as e:
            print("Failed to save suspicious snapshot:", e)
    # submit copy to avoid races
    executor.submit(_save, crop.copy(), tid)
# -------------------------------------------------------------------

# ----------------- Video IO -----------------
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open video source."

# if using webcam and we resize for inference, preserve original output size for saving
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
# frames required to count as STACK_SECONDS_TO_INCREMENT seconds
frames_for_stack_inc = max(1, int(ROUND := int(STACK_SECONDS_TO_INCREMENT * fps)))  # used below

out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

frame_idx = 0
next_id = 1
prev_centroids = {}

# store last detections (to reuse when skipping frames)
_last_dets_for_tracking = []

print("Starting optimized main loop... Press 'q' to quit.")

# ---------------- main loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    if camera_obstructed(frame):
        cv2.putText(frame, "ALERT: CAMERA OBSTRUCTED", (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        out.write(frame)
        if not DISABLE_PREVIEW:  # ✅ integrated disable preview toggle
            cv2.imshow("CamSentinel Live", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        continue
        
    # optionally resize for inference (we will map coords back)
    scale_x = scale_y = 1.0
    inference_frame = frame
    if RESIZE_WH is not None:
        rw, rh = RESIZE_WH
        ih, iw = frame.shape[:2]
        scale_x = iw / float(rw)
        scale_y = ih / float(rh)
        inference_frame = cv2.resize(frame, (rw, rh))

    # decide if we run detection this frame or reuse last detections
    run_detection = (frame_idx % max(1, DETECT_EVERY_N) == 0)

    dets = []
    if run_detection:
        # run YOLO on inference_frame
        results = model(inference_frame, verbose=False)[0]
        for box in results.boxes:
            try:
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
            except Exception:
                # older / different ultralytics API safety
                conf = float(box.conf)
                cls = int(box.cls)
            if conf < MIN_CONF or cls in IGNORE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # map coords back to original frame size
            if RESIZE_WH is not None:
                x1_o = int(x1 * scale_x)
                y1_o = int(y1 * scale_y)
                x2_o = int(x2 * scale_x)
                y2_o = int(y2 * scale_y)
            else:
                x1_o, y1_o, x2_o, y2_o = x1, y1, x2, y2

            # width/height ratio filter (ignore too-wide)
            if (x2_o - x1_o) / (max(1, (y2_o - y1_o))) > MAX_ASPECT_RATIO:
                continue

            dets.append([x1_o, y1_o, x2_o, y2_o, conf, cls])

        # store last detections for reuse while skipping frames
        _last_dets_for_tracking = dets.copy()
    else:
        # reuse last detections (so tracker continues to be fed)
        dets = _last_dets_for_tracking.copy()

    # crowd filtering (unchanged)
    filtered_dets = []
    for i, d in enumerate(dets):
        overlap = False
        for j, d2 in enumerate(dets):
            if i == j: continue
            if iou(d[:4], d2[:4]) > IOU_FILTER_THRESHOLD:
                if d[4] < d2[4]:
                    overlap = True
                    break
        if not overlap:
            filtered_dets.append(d)
    dets = filtered_dets

    # tracking (unchanged)
    tracked_outs = []
    if use_deepsort and tracker:
        ds_in = [([d[0], d[1], d[2]-d[0], d[3]-d[1]], d[4], str(d[5])) for d in dets]
        tracks = tracker.update_tracks(ds_in, frame=frame)
        for t in tracks:
            try:
                if not t.is_confirmed(): continue
                x1, y1, x2, y2 = map(int, t.to_ltrb())
                tracked_outs.append((t.track_id, x1, y1, x2, y2, t.get_det_conf()))
            except Exception:
                pass
    else:
        # centroid fallback (unchanged)
        cur_centroids = []
        for d in dets:
            x1, y1, x2, y2, conf, cls = d
            cx, cy = ((x1+x2)/2, (y1+y2)/2)
            cur_centroids.append((cx, cy, x1, y1, x2, y2, conf))
        assigned = set()
        new_prev = {}
        for tid, pc in list(prev_centroids.items()):
            best_d, bi = 1e9, None
            for i, c in enumerate(cur_centroids):
                if i in assigned: continue
                d = (pc[0]-c[0])**2 + (pc[1]-c[1])**2
                if d < best_d: best_d, bi = d, i
            if bi is not None and best_d < 3600:
                cx, cy, x1, y1, x2, y2, conf = cur_centroids[bi]
                tracked_outs.append((tid, x1, y1, x2, y2, conf))
                assigned.add(bi)
                new_prev[tid] = (cx, cy)
        for i, c in enumerate(cur_centroids):
            if i not in assigned:
                tid = next_id; next_id += 1
                cx, cy, x1, y1, x2, y2, conf = c
                tracked_outs.append((tid, x1, y1, x2, y2, conf))
                new_prev[tid] = (cx, cy)
        prev_centroids = new_prev.copy()

    # main per-track logic (mostly unchanged; augmented with async embedding & stacks)
    for (tid, x1, y1, x2, y2, conf) in tracked_outs:
        cx, cy = bbox_centroid(x1, y1, x2, y2)
        centroid_history[tid].append((cx, cy))
        last_seen_frame[tid] = frame_idx
        recent_boxes[tid].append((frame_idx, (x1, y1, x2, y2)))

        disp = 0
        if len(centroid_history[tid]) > 1:
            (ox, oy) = centroid_history[tid][0]
            (nx, ny) = centroid_history[tid][-1]
            disp = np.hypot(nx-ox, ny-oy)
        stationary_flag[tid] = (len(centroid_history[tid]) == centroid_history[tid].maxlen and disp < DISPLACEMENT_THRESHOLD)

        name = None
        # safety checks for crop bounds
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2c <= x1c or y2c <= y1c:
            crop = None
        else:
            crop = frame[y1c:y2c, x1c:x2c]

        # Face recognition: try FR first (sync) then facenet async fallback
        feat = None
        feat_type = None

        # ✅ Skip recognition/tracking work early for known/persistent or permanent suspicious
        # (saves CPU by not computing embeddings or updating persistence logic further)
        if tid in known_persistent_ids or tid in permanent_suspicious:
            # if we have a name already, prefer that. We will still draw bounding boxes below.
            name = known_persistence.get(tid, {}).get("name", name)
            feat = None
            feat_type = None
        else:
            if crop is not None and len(known_face_entries) > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                # face_recognition (synchronous)
                if FACE_RECOG_AVAILABLE:
                    try:
                        encs = face_recognition.face_encodings(rgb)
                        if len(encs) > 0:
                            feat = np.asarray(encs[0], dtype=np.float32)
                            feat_type = 'fr'
                    except Exception:
                        pass
                # if no FR embedding, schedule / use facenet embeddings (async)
                if feat is None and FACENET_AVAILABLE:
                    # check if we already have a cached embedding for this tid and it's recent enough
                    last_emb = last_embedding.get(tid)
                    last_emb_fr = last_embedding_frame.get(tid, -9999)
                    # if cached embedding exists and is recent enough (we consider it usable)
                    if last_emb is not None and (frame_idx - last_emb_fr) < FACENET_EVERY_N:
                        feat = last_emb
                        feat_type = 'facenet'
                    else:
                        # schedule an async embedding if allowed
                        # we pass the RGB crop to the background worker
                        schedule_embedding_for_tid(tid, rgb, frame_idx)
                        # if there's a completed future already, grab it
                        fut = embedding_future.get(tid)
                        if fut is not None and fut.done():
                            try:
                                emb_val = fut.result()
                                last_embedding[tid] = emb_val.astype(np.float32)
                                last_embedding_frame[tid] = frame_idx
                                feat = last_embedding[tid]
                                feat_type = 'facenet'
                            except Exception:
                                feat = None

                # If we have a feat (either FR or facenet cached), do matching
                if feat is not None and feat_type is not None:
                    best_name, best_score = None, 1e9
                    for ref_enc, ref_name, ref_type in known_face_entries:
                        if ref_type != feat_type: continue
                        score = face_distance_euclidean(ref_enc, feat) if feat_type == 'fr' else face_distance_cosine(ref_enc, feat)
                        if score < best_score:
                            best_score, best_name = score, ref_name
                    if (feat_type == 'fr' and best_score < FR_EUCLIDEAN_THRESHOLD) or (feat_type == 'facenet' and best_score < FACENET_COSINE_THRESHOLD):
                        name = best_name

        # persistence logic (unchanged)
        if tid in known_persistent_ids:
            name = known_persistence[tid].get("name", name)
        if name:
            st = known_persistence[tid]
            st["name"] = name
            st["consec"] += 1
            st["total"] += 1
            if st["consec"] >= KNOWN_PERSIST_CONSEC or st["total"] >= KNOWN_PERSIST_NONCONSEC:
                known_persistent_ids.add(tid)
        else:
            known_persistence[tid]["consec"] = 0

        # ---------------- Suspicious stacks logic ----------------
        # Behavior:
        # - On first time stationary => stack = 1, save snapshot
        # - If remains stationary and STACK_SECONDS_TO_INCREMENT elapsed => auto-increment stack
        # - If stack reaches STACK_LIMIT => mark as permanently suspicious and stop heavy processing thereafter
        if stationary_flag.get(tid, False) and tid not in known_persistent_ids and tid not in permanent_suspicious:
            # if first time stacking for this tid
            if suspicious_stacks[tid] == 0:
                suspicious_stacks[tid] = 1
                stack_last_inc_frame[tid] = frame_idx
                # save snapshot for stack 1
                if crop is not None:
                    save_suspicious_snapshot_async(crop, tid)
                    print(f"🔶 Tid {tid} -> Stack 1 (first suspicious snapshot saved)")
            else:
                # if stationary persists long enough since last increment, increment stack
                last_inc = stack_last_inc_frame.get(tid, -999999)
                if (frame_idx - last_inc) >= max(1, int(STACK_SECONDS_TO_INCREMENT * fps)):
                    if suspicious_stacks[tid] < STACK_LIMIT:
                        suspicious_stacks[tid] += 1
                        stack_last_inc_frame[tid] = frame_idx
                        # save snapshot for this new stack increment
                        if crop is not None:
                            save_suspicious_snapshot_async(crop, tid)
                        print(f"🔶 Tid {tid} -> Stack {suspicious_stacks[tid]} (auto-increment)")
                        if suspicious_stacks[tid] >= STACK_LIMIT:
                            permanent_suspicious.add(tid)
                            print(f"⚠️ Person ID {tid} is now permanently suspicious (stack={STACK_LIMIT})")
        # Note: we intentionally do NOT decrement stacks if person moves away. Stacks persist (as requested).

        # color logic (unchanged but extended for stacks/permanent)
        if tid in known_persistent_ids or name:
            color = (255, 255, 0)
        elif tid in permanent_suspicious:
            color = (0, 0, 150)  # deep red for permanently suspicious
        elif stationary_flag.get(tid, False):
            # progressively darker red per stack
            stack_c = suspicious_stacks.get(tid, 0)
            color = (0, 0, min(255, 100 + stack_c * 50))
        else:
            color = (0, 255, 0)

        label = name if (name or tid in known_persistent_ids) else f"ID {tid}"
        if tid in suspicious_stacks and suspicious_stacks[tid] > 0:
            label += f" [Stack {suspicious_stacks[tid]}]"  # ✅ show stack count

        # draw rectangle & label (restored so code is complete and runnable)
        try:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (max(0, x1), max(10, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            # in case coordinates are malformed, skip drawing rather than crashing
            pass

    # draw recent boxes (unchanged)
    for tid, boxes in recent_boxes.items():
        for f_idx, (x1, y1, x2, y2) in list(boxes):
            if frame_idx - f_idx <= SHOW_BOX_LAST_FRAMES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # display + write + FPS sync (unchanged)
    if not DISABLE_PREVIEW:  # ✅ preview toggle integration
        cv2.imshow("CamSentinel Live", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            print("Exiting on user request (q pressed).")
            break

    if not SKIP_FRAMES_FOR_WRITING:
        out.write(frame)

# cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
print("✅ Finished. Output saved to", VIDEO_OUT)
