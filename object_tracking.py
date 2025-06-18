"""
Object tracking – YOLOv11 + Deep‑SORT
------------------------------------
• Chỉ track class 53  (tracking_class = None  ➞ track tất cả)
• Nhấn Q để thoát.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort   # pip install deep‑sort‑realtime

# ------------- CONFIG ------------------------------------------------------
VIDEO_PATH      = r"SOHO_train\SOHO\1467_CH04_20250607180000_190000.mp4"
WEIGHTS_PATH    = r"C:\Users\Admin\Downloads\Pizza\yolo11x.pt"       # chỉnh đúng đường dẫn
CONF_THRESHOLD  = 0.5
TRACKING_CLASS  = 53                           # None → track all
DEVICE          = 0 if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------

# 1️⃣ Load YOLO (class mới đã gộp backend + autoshape)
model = YOLO(WEIGHTS_PATH).to(DEVICE)

# 2️⃣ Deep‑SORT tracker
tracker = DeepSort(max_age=100000)

# 3️⃣ Class names & màu
with open("data_ext/classes.names") as f:
    CLASS_NAMES = f.read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3))

# 4️⃣ Đọc video
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open: {VIDEO_PATH}"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── 4a. YOLO detect ────────────────────────────────────────────────────
    results = model(frame, verbose=False)[0]             # Results obj

    detections = []
    # chuyển tensor → numpy cho Deep‑SORT
    for xyxy, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                               results.boxes.conf.cpu().numpy(),
                               results.boxes.cls.cpu().numpy()):
        class_id = int(cls)
        if TRACKING_CLASS is not None and class_id != TRACKING_CLASS:
            continue
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = xyxy.astype(int)
        w, h = x2 - x1, y2 - y1
        detections.append([[x1, y1, w, h], conf, class_id])

    # ── 4b. Deep‑SORT update ──────────────────────────────────────────────
    tracks = tracker.update_tracks(detections, frame=frame)

    # ── 4c. Vẽ kết quả ────────────────────────────────────────────────────
    for trk in tracks:
        if not trk.is_confirmed():
            continue
        track_id  = trk.track_id
        x1, y1, x2, y2 = map(int, trk.to_ltrb())
        class_id  = trk.get_det_class() or 0

        color = COLORS[class_id].tolist()
        label = f"{CLASS_NAMES[class_id]}-{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + len(label) * 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv11 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
