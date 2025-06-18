import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO(r"runs\detect\yolov8_pizza_detection\weights\best.pt")
tracker = DeepSort(max_age=30)
video_path = r"SOHO_train\SOHO\1462_CH03_20250607192844_202844.mp4"
cap = cv2.VideoCapture(video_path)

frame_skip = 150
frame_count = 0
target_class = 53
conf_threshold = 0.5
line_y = 200
counted_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # if frame_count % frame_skip != 0:
    #     continue

    detections = []
    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == target_class and conf > conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cy = (y1 + y2) // 2
        if tid not in counted_ids and cy > line_y:
            counted_ids.add(tid)

        # Vẽ bounding box và ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Vẽ line đếm và số lượng
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Pizza Count: {len(counted_ids)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Hiển thị frame
    cv2.imshow("Pizza Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2, time
# import numpy as np
# import os

# # ---------- 1) Mở video ----------
# VIDEO = r"SOHO_train\SOHO\1462_CH03_20250607192844_202844.mp4"
# if not os.path.exists(VIDEO):
#     raise FileNotFoundError(f"Không tìm thấy video: {VIDEO}")

# cap = cv2.VideoCapture(VIDEO)
# ret, first_frame = cap.read()
# if not ret:
#     raise RuntimeError("Không đọc được frame đầu!")

# # ---------- 2) Chọn ROI bằng chuột (kéo‑thả) ----------
# print("Kéo & thả để chọn ROI đáy hộp, ENTER để chốt, ESC để huỷ")
# roi_box = cv2.selectROI("Chọn ROI", first_frame, showCrosshair=False)
# cv2.destroyWindow("Chọn ROI")

# x, y, w, h = map(int, roi_box)
# print("ROI đã chọn:", (x, y, w, h))

# # ---------- 3) Thông số & biến trạng thái ----------
# DIFF_TH      =  800   # Ngưỡng MSE > DIFF_TH ⇒ có pizza
# HOLD_MIN     = 0.3    # (giây) pizza phải “ở yên” mới tính là FILLED
# REFRESH_BASE = False  # True nếu muốn baseline reset mỗi lần đếm

# baseline, state = None, "EMPTY"
# count, last_filled_ts = 0, 0

# fps = cap.get(cv2.CAP_PROP_FPS)
# fps = fps if fps > 1 else 25       # fallback nếu video không ghi FPS

# # ---------- 4) Hàm đo MSE ----------
# def mse(img1, img2):
#     return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

# # ---------- 5) Vòng lặp ----------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     roi = frame[y:y+h, x:x+w]
#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

#     if baseline is None:
#         baseline = roi_gray.copy()
#         continue

#     diff_score = mse(baseline, roi_gray)

#     # ------- FSM --------
#     if state == "EMPTY" and diff_score > DIFF_TH:
#         state = "FILLED"
#         last_filled_ts = time.time()

#     elif state == "FILLED":
#         stay_time = time.time() - last_filled_ts
#         # khi ROI giống lại baseline + pizza đã ở ≥ HOLD_MIN giây
#         if diff_score <= DIFF_TH and stay_time >= HOLD_MIN:
#             state = "EMPTY"
#             count += 1
#             print(f"[{time.strftime('%H:%M:%S')}] +1 Pizza  →  Tổng: {count}")
#             if REFRESH_BASE:
#                 baseline = roi_gray.copy()   # cập nhật baseline mới

#     # ------- Vẽ & hiển thị --------
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#     cv2.putText(frame, f"Count: {count}", (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
#     cv2.imshow("Pizza Counter (q to quit)", frame)

#     if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
