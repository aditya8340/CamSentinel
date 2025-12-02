






# ==========================================
# YOLOv8 + DeepSORT + Stationary Detection + Camera Obstruction + Face Recognition (One-shot)
# ==========================================

# 1) Install dependencies
# !pip install ultralytics opencv-python-headless==4.9.0.80 filterpy scipy --quiet
# !pip install deep_sort_realtime --quiet || true
# !pip install face_recognition --quiet dlib --quiet

# 2) Imports
import cv2, os
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import face_recognition

# Try to import deep-sort
use_deepsort = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    use_deepsort = True
    print("✅ Using deep_sort_realtime for DeepSORT-style tracking.")
except Exception as e:
    print("⚠️ DeepSORT not available, fallback to centroid-based tracking. Error:", e)

# 3) Parameters
VIDEO_IN = "stationary check.mp4"
VIDEO_OUT = "full_system_output.mp4"
MIN_CONF = 0.27
STATIONARY_FRAMES = 60                # ~2s at 30fps
DISPLACEMENT_THRESHOLD = 10           # pixels
TARGET_CLASSES = [0, 2]               # 0=person, 2=car

# 4) Initialize YOLO
model = YOLO("yolov8n.pt")

# 5) Initialize tracker
tracker = None
if use_deepsort:
    tracker = DeepSort(max_age=30)

# 6) Stationary bookkeeping
centroid_history = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
last_seen_frame = {}
stationary_flag = {}

# 7) Load reference faces (database for one-shot recognition)
# inside your main script (where load_reference_faces was)
from face_utils import pil_load_and_fix, safe_face_encodings_from_array
import os
known_face_encodings = []
known_face_names = []

def load_reference_faces(ref_dir="faces_db", max_dim=1200):
    """
    Load images under faces_db/{person}/... and compute encodings robustly.
    Saves problematic file names to console if encoding fails.
    """
    if not os.path.exists(ref_dir):
        print("No faces_db folder found -> skipping face db load.")
        return

    for person_name in os.listdir(ref_dir):
        person_folder = os.path.join(ref_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        for fname in os.listdir(person_folder):
            path = os.path.join(person_folder, fname)
            try:
                img = pil_load_and_fix(path, max_dim=max_dim)  # RGB uint8
                encs = safe_face_encodings_from_array(img, from_bgr=False)
                if len(encs) > 0:
                    known_face_encodings.append(encs[0])
                    known_face_names.append(person_name)
                    print(f"Loaded face for {person_name} from {fname}")
                else:
                    print(f"❗ No faces found in {path} (skipped).")
            except Exception as e:
                print(f"Error loading face {path}: {e}")


# Example: provide your database images
if os.path.exists("faces_db"):
    load_reference_faces("faces_db")
else:
    print("⚠️ No faces_db folder found. Face recognition disabled.")

# Helper: compute centroid
def bbox_centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

# Helper: detect camera obstruction
def camera_obstructed(frame, brightness_thresh=15, variance_thresh=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    var_val = np.var(gray)
    if mean_val < brightness_thresh or mean_val > 255 - brightness_thresh:
        return True
    if var_val < variance_thresh:
        return True
    return False

# 8) Open video
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open input video."

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

frame_idx = 0
if not use_deepsort:
    fallback_tracks = {}
    next_id = 1
    prev_centroids = {}

# 9) Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # --- Camera obstruction check ---
    if camera_obstructed(frame):
        cv2.putText(frame, "🚨 ALERT: CAMERA OBSTRUCTED", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        out.write(frame)
        continue

    # --- YOLO detection ---
    results = model(frame, verbose=False)[0]

    dets = []
    for box in results.boxes:
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        if conf < MIN_CONF or cls not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        dets.append([float(x1), float(y1), float(x2), float(y2), conf, cls])

    tracked_outs = []

    if use_deepsort and tracker is not None:
        ds_input = []
        for d in dets:
            x1,y1,x2,y2,score,cls = d
            ds_input.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], score, str(cls)))
        tracks = tracker.update_tracks(ds_input, frame=frame)
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = t.to_ltrb()
            det_class = getattr(t, 'det_class', None)
            score = getattr(t, 'det_conf', None)
            tracked_outs.append((tid, float(x1), float(y1), float(x2), float(y2), det_class, score))

    else:
        cur_centroids = []
        for i,d in enumerate(dets):
            x1,y1,x2,y2,score,cls = d
            cx,cy = bbox_centroid(x1,y1,x2,y2)
            cur_centroids.append((cx,cy,x1,y1,x2,y2,score,cls))

        assigned = set()
        new_prev = {}
        for tid, pc in list(prev_centroids.items()):
            best_i = None
            best_dist = 1e9
            for idx, (cx,cy, *_) in enumerate(cur_centroids):
                if idx in assigned: continue
                dist = (pc[0]-cx)**2 + (pc[1]-cy)**2
                if dist < best_dist:
                    best_dist = dist
                    best_i = idx
            if best_i is not None and best_dist < (50*50):
                cx,cy,x1,y1,x2,y2,score,cls = cur_centroids[best_i]
                tracked_outs.append((tid, x1,y1,x2,y2, cls, score))
                assigned.add(best_i)
                new_prev[tid] = (cx,cy)
        for idx, (cx,cy,x1,y1,x2,y2,score,cls) in enumerate(cur_centroids):
            if idx in assigned: continue
            tid = next_id
            next_id += 1
            tracked_outs.append((tid, x1,y1,x2,y2, cls, score))
            new_prev[tid] = (cx,cy)
        prev_centroids = new_prev.copy()

    # --- Stationary + Face Recognition ---
    for (tid, x1, y1, x2, y2, cls, score) in tracked_outs:
        cx, cy = bbox_centroid(x1, y1, x2, y2)
        centroid_history[tid].append((cx, cy, frame_idx))
        last_seen_frame[tid] = frame_idx

        disp = float('inf')
        hist = centroid_history[tid]
        if len(hist) >= 2:
            old_cx, old_cy, _ = hist[0]
            new_cx, new_cy, _ = hist[-1]
            disp = np.hypot(new_cx - old_cx, new_cy - old_cy)

        is_stationary = (len(hist) == hist.maxlen) and (disp < DISPLACEMENT_THRESHOLD)
        prev_flag = stationary_flag.get(tid, False)
        if is_stationary and not prev_flag:
            print(f"🚨 ALERT: Track {tid} (cls {cls}) stationary at frame {frame_idx}, disp={disp:.2f}")
            stationary_flag[tid] = True
        elif not is_stationary and prev_flag:
            print(f"ℹ️ INFO: Track {tid} moved again at frame {frame_idx}.")
            stationary_flag[tid] = False

        # Face recognition only if person (cls=0)
        person_name = None
        if cls == 0 and len(known_face_encodings) > 0:
            face_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_face)
            if len(encs) > 0:
                face_enc = encs[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=0.45)
                face_dists = face_recognition.face_distance(known_face_encodings, face_enc)
                if True in matches:
                    best_idx = np.argmin(face_dists)
                    person_name = known_face_names[best_idx]

        # Draw bounding box + label
        if stationary_flag.get(tid, False):
            color = (0,0,255)
            base_label = f"ID {tid} STATIONARY"
        else:
            color = (0,255,0)
            base_label = f"ID {tid}"

        if person_name:
            label = f"{person_name}"
        else:
            label = base_label

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # cleanup old tracks
    to_delete = []
    for tid, last in list(last_seen_frame.items()):
        if frame_idx - last > (fps * 5):
            to_delete.append(tid)
    for tid in to_delete:
        centroid_history.pop(tid, None)
        last_seen_frame.pop(tid, None)
        stationary_flag.pop(tid, None)

    out.write(frame)

cap.release()
out.release()
print("✅ Finished. Output saved to", VIDEO_OUT)
