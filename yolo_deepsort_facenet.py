"""
YOLOv8 + DeepSORT (or centroid fallback) + Stationary check + Camera-obstruct +
Face ID (face_recognition fallback -> facenet-pytorch) + persistence + suspicious capture +
timestamped snapshots + live preview + FPS sync + wide-object filter

Minimal CLI addition: accept an optional command-line argument for VIDEO_IN.
"""
import os
import cv2
import numpy as np
import sys                          # <-- added for CLI parsing
from collections import defaultdict, deque
from datetime import datetime  # ⏰ timestamp
from ultralytics import YOLO

# ---------------- CLI parsing (minimal, safe) ----------------
# If a command-line argument is provided, use it as VIDEO_IN.
# If it's convertible to int, treat as camera index (e.g. 0). Otherwise treat as file path.
_cmd_video_in = None
if len(sys.argv) > 1:
    _arg = sys.argv[1]
    try:
        _cmd_video_in = int(_arg)
    except Exception:
        _cmd_video_in = _arg
# ------------------------------------------------------------

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

# ---------------- PARAMETERS ----------------
# NOTE: VIDEO_IN will use CLI argument if provided; else default to 0 (webcam).
VIDEO_IN = _cmd_video_in if _cmd_video_in is not None else 0  # Change to "video.mp4" for recorded file
VIDEO_OUT = "full_system_facenet_corrected_final_2_3.mp4"
MIN_CONF = 0.35
STATIONARY_FRAMES = 60
DISPLACEMENT_THRESHOLD = 10
TARGET_CLASSES = [0]  # person
IGNORE_CLASSES = [39, 41, 67]
FR_EUCLIDEAN_THRESHOLD = 0.47 # 47 to 50 to 48 to 47
FACENET_COSINE_THRESHOLD = 0.20 # 22 to 25 to 20
IOU_FILTER_THRESHOLD = 0.6
MAX_ASPECT_RATIO = 1.2  # ✅ width/height filter

# ---------------- Behavior persistence & snapshot ----------------
KNOWN_PERSIST_CONSEC = 15
KNOWN_PERSIST_NONCONSEC = 30
SUSPICIOUS_SAVE_FOLDER = "suspicious_person"
SHOW_BOX_LAST_FRAMES = 1
# --------------------------------------------

os.makedirs(SUSPICIOUS_SAVE_FOLDER, exist_ok=True)
model = YOLO("yolov8n.pt")

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

def bbox_centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

# -------------- face DB --------------
known_face_entries = []

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

def encode_with_facenet(img_rgb):
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

def load_face_db(folder="faces_db"):
    if not os.path.exists(folder):
        print("No face DB folder:", folder)
        return
    for person in sorted(os.listdir(folder)):
        pdir = os.path.join(folder, person)
        if not os.path.isdir(pdir): continue
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
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
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(area1 + area2 - inter + 1e-6)

# --------------- main ----------------
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open video source."

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_idx = 0
next_id = 1
prev_centroids = {}

print("Starting main loop... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    if camera_obstructed(frame):
        cv2.putText(frame, "ALERT: CAMERA OBSTRUCTED", (40,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        out.write(frame)
        continue

    results = model(frame, verbose=False)[0]
    dets = []
    for box in results.boxes:
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        if conf < MIN_CONF or cls in IGNORE_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # ✅ Width-height ratio filter (ignore too-wide non-human shapes)
        if (x2 - x1) / (y2 - y1 + 1e-5) > MAX_ASPECT_RATIO:
            continue

        dets.append([x1, y1, x2, y2, conf, cls])

    # crowd filtering
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

    # tracking
    tracked_outs = []
    if use_deepsort and tracker:
        ds_in = [([d[0], d[1], d[2]-d[0], d[3]-d[1]], d[4], str(d[5])) for d in dets]
        tracks = tracker.update_tracks(ds_in, frame=frame)
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            tracked_outs.append((t.track_id, x1, y1, x2, y2, t.get_det_conf()))
    else:
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
                if i in assigned:
                    continue
                d = (pc[0]-c[0])**2 + (pc[1]-c[1])**2
                if d < best_d:
                    best_d, bi = d, i
            if bi is not None and best_d < 3600:
                cx, cy, x1, y1, x2, y2, conf = cur_centroids[bi]
                tracked_outs.append((tid, x1, y1, x2, y2, conf))
                assigned.add(bi)
                new_prev[tid] = (cx, cy)
        for i, c in enumerate(cur_centroids):
            if i not in assigned:
                tid = next_id
                next_id += 1
                cx, cy, x1, y1, x2, y2, conf = c
                tracked_outs.append((tid, x1, y1, x2, y2, conf))
                new_prev[tid] = (cx, cy)
        prev_centroids = new_prev.copy()

    # main tracking loop
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
        stationary_flag[tid] = (
            len(centroid_history[tid]) == centroid_history[tid].maxlen
            and disp < DISPLACEMENT_THRESHOLD
        )

        name = None
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0 and len(known_face_entries) > 0:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            feat, feat_type = None, None
            if FACE_RECOG_AVAILABLE:
                try:
                    encs = face_recognition.face_encodings(rgb)
                    if len(encs) > 0:
                        feat = np.asarray(encs[0], dtype=np.float32)
                        feat_type = 'fr'
                except Exception:
                    pass
            if feat is None and FACENET_AVAILABLE:
                try:
                    feat = encode_with_facenet(rgb).astype(np.float32)
                    feat_type = 'facenet'
                except Exception:
                    feat = None

            if feat is not None:
                best_name, best_score = None, 1e9
                for ref_enc, ref_name, ref_type in known_face_entries:
                    if ref_type != feat_type:
                        continue
                    score = (
                        face_distance_euclidean(ref_enc, feat)
                        if feat_type == "fr"
                        else face_distance_cosine(ref_enc, feat)
                    )
                    if score < best_score:
                        best_score, best_name = score, ref_name
                if (feat_type == "fr" and best_score < FR_EUCLIDEAN_THRESHOLD) or (
                    feat_type == "facenet" and best_score < FACENET_COSINE_THRESHOLD
                ):
                    name = best_name

        # persistence logic
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

        # snapshot suspicious (timestamped)
        if stationary_flag.get(tid, False) and tid not in saved_suspicious and tid not in known_persistent_ids:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(SUSPICIOUS_SAVE_FOLDER, f"suspicious_{tid}_{timestamp}.jpg")
            cv2.imwrite(snap_path, crop)
            saved_suspicious.add(tid)
            print(f"📸 Saved suspicious snapshot {snap_path}")

        # color logic
        if tid in known_persistent_ids or name:
            color = (255, 255, 0)
        elif stationary_flag.get(tid, False):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        label = name if (name or tid in known_persistent_ids) else f"ID {tid}"
        cv2.putText(frame, label, (max(0, x1), max(10, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # draw recent boxes
    for tid, boxes in recent_boxes.items():
        for f_idx, (x1, y1, x2, y2) in list(boxes):
            if frame_idx - f_idx <= SHOW_BOX_LAST_FRAMES:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # ✅ display + write + FPS sync
    cv2.imshow("CamSentinel Live", frame)
    out.write(frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        print("Exiting on user request (q pressed).")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Finished. Output saved to", VIDEO_OUT)
