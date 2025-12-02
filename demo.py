# ==========================================
# YOLOv8 + DeepSORT tracking with Stationary Detection
# ==========================================

# 1) Install dependencies
# !pip install ultralytics opencv-python-headless==4.9.0.80 filterpy scipy --quiet
# !pip install deep_sort_realtime --quiet || true

# 2) Imports
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

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
VIDEO_OUT = "deepsort_tracked_output_27.mp4"
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

# Helper: compute centroid
def bbox_centroid(x1, y1, x2, y2):
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

# 7) Open video
cap = cv2.VideoCapture(VIDEO_IN)
assert cap.isOpened(), "Cannot open input video."

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

frame_idx = 0

# If fallback mode, prepare structures
if not use_deepsort:
    fallback_tracks = {}
    next_id = 1
    prev_centroids = {}

# 8) Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame, verbose=False)[0]

    # Collect detections
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
        # DeepSORT expects tlwh
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
        # Fallback: greedy centroid matching
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

    # Update stationary detection
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

        # Draw bounding box
        if stationary_flag.get(tid, False):
            c1 = (0, 0, 255)
            label = f"ID {int(tid)} cls:{cls} STATIONARY"
        else:
            c1 = (0, 255, 0)
            label = f"ID {int(tid)} cls:{cls}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c1, 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 2)

    # Cleanup old tracks
    to_delete = []
    for tid, last in list(last_seen_frame.items()):
        if frame_idx - last > (fps * 5):
            to_delete.append(tid)
    for tid in to_delete:
        centroid_history.pop(tid, None)
        last_seen_frame.pop(tid, None)
        stationary_flag.pop(tid, None)

    out.write(frame)

# release
cap.release()
out.release()
print("✅ Finished. Output saved to", VIDEO_OUT)