"""python
pizza_counter_pipeline.py – Interactive Region Counting Pipeline
===============================================================
Video ➜ Object Detection (YOLOv8) ➜ Object Tracking (DeepSORT) ➜ **User‑defined Polygon ROI** ➜ Count ➜ Output

How to use
----------
1. Install dependencies:
   pip install ultralytics deep_sort_realtime opencv-python numpy shapely

2. Run script:
   python pizza_counter_pipeline.py --video path/to/video.mp4 --weights yolov8n.pt --classes 53 --device 0

3. Configuration window opens on first frame:
   • **Drag vertices** (yellow) to reshape the polygon
   • **Drag inside polygon** to move whole region
   • Press **ENTER** to start processing, or **q** to quit

4. Processing window shows live bounding‑boxes & running pizza count.
   Press **q** anytime to stop.
"""
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ------------------------ Global state for interactive ROI ------------------------
REGION_COLOR = (0, 0, 255)      # Red polygon
REGION_TEXT_COLOR = (255, 255, 255)
vertex_radius = 6

polygon_roi: Polygon | None = None        # will hold polygon after config
cur_vertex = -1                           # index of vertex being dragged
cur_offset = (0, 0)                       # offset when dragging polygon
is_drag_poly = False


def mouse_callback(event, x, y, flags, param):
    global polygon_roi, cur_vertex, is_drag_poly, cur_offset

    if polygon_roi is None:
        return

    pts = list(polygon_roi.exterior.coords)[:-1]  # shapely repeats first point; drop last
    p = Point((x, y))

    if event == cv2.EVENT_LBUTTONDOWN:
        # check vertices first
        for i, (vx, vy) in enumerate(pts):
            if Point((vx, vy)).distance(p) < 12:
                cur_vertex = i
                return
        # else if click inside polygon -> drag whole poly
        if polygon_roi.contains(p):
            is_drag_poly = True
            cur_offset = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if cur_vertex != -1:
            pts[cur_vertex] = (x, y)
            polygon_roi = Polygon(pts)
        elif is_drag_poly:
            dx, dy = x - cur_offset[0], y - cur_offset[1]
            pts = [(px + dx, py + dy) for (px, py) in pts]
            polygon_roi = Polygon(pts)
            cur_offset = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        cur_vertex = -1
        is_drag_poly = False


# ------------------------ helper functions ------------------------

def draw_polygon(frame):
    if polygon_roi is None:
        return frame
    pts = np.array(list(polygon_roi.exterior.coords), np.int32)
    cv2.polylines(frame, [pts], True, REGION_COLOR, 2)
    # draw vertices
    for (vx, vy) in pts[:-1]:
        cv2.circle(frame, (int(vx), int(vy)), vertex_radius, (0, 255, 255), -1)
    # label
    cx, cy = int(polygon_roi.centroid.x), int(polygon_roi.centroid.y)
    cv2.putText(frame, "ROI", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, REGION_TEXT_COLOR, 2)
    return frame


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='Path to input video')
    ap.add_argument('--weights', default='yolov8n.pt', help='YOLO checkpoint')
    ap.add_argument('--classes', nargs='+', type=int, default=[53], help='Class IDs to detect')
    ap.add_argument('--device', default='0', help='cuda device or cpu')
    ap.add_argument('--out', default='output.mp4', help='output video file')
    ap.add_argument('--skip', type=int, default=1, help='process every Nth frame')
    return ap.parse_args()


def main():
    args = parse_args()

    # -------- load video & model --------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                           cv2.CAP_PROP_FRAME_HEIGHT,
                                           cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.weights)
    model.to('cuda:0' if args.device == '0' else 'cpu')
    tracker = DeepSort(max_age=30, n_init=2)

    # -------- configuration window --------
    ret, first = cap.read()
    if not ret:
        print('Cannot read first frame'); return

    # default polygon (rectangle 1/3 center)
    global polygon_roi
    polygon_roi = Polygon([(w*0.3, h*0.3), (w*0.7, h*0.3), (w*0.7, h*0.7), (w*0.3, h*0.7)])

    cv2.namedWindow('Configure ROI')
    cv2.setMouseCallback('Configure ROI', mouse_callback)
    print('Drag yellow points to reshape ROI, drag inside polygon to move. Press ENTER when done.')

    while True:
        disp = draw_polygon(first.copy())
        cv2.imshow('Configure ROI', disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # ENTER
            cv2.destroyWindow('Configure ROI')
            break
        elif k == ord('q'):
            return

    # -------- processing loop --------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    counted_ids = set()
    prev_center = {}
    total = 0
    frame_id = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % args.skip:
            writer.write(frame)
            continue

        # detection
        det = model(frame, conf=0.4, verbose=False, classes=args.classes)[0]
        detections = []
        for b in det.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            detections.append(([x1, y1, x2-x1, y2-y1], float(b.conf[0]), int(b.cls[0])))

        # tracking
        tracks = tracker.update_tracks(detections, frame=frame)
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            if polygon_roi.contains(Point((cx, cy))) and tid not in counted_ids:
                total += 1
                counted_ids.add(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID{tid}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        frame = draw_polygon(frame)
        cv2.putText(frame, f'Count: {total}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3)

        writer.write(frame)
        cv2.imshow('Pizza Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start
    print(f'Total pizzas counted: {total}')
    print(f'Processing FPS: {frame_id/elapsed:.2f}')


if __name__ == '__main__':
    main()
