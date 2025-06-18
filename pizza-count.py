from ultralytics import solutions
import cv2  

cap = cv2.VideoCapture(r"SOHO_train\SOHO\1465_CH02_20250607170555_172408.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("counting.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

region_points = [(75, 628), (1449, 636)] 

# Init ObjectCounter
counter =solutions.RegionCounter(
    show=True,  # Display the outputqqqqq 
    region=region_points,  # Pass region points
    model=r"yolo12x.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    classes=[53],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    # show_in=True,  # Display in counts
    # show_out=True,  # Display out counts
    line_width=3,  # Adjust the line width for bounding boxes and text display
)
# Thêm biến đếm khung hình ở đây
frame_count = 0
# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # Tăng bộ đếm khung hình
    frame_count += 1
    
    # Bỏ qua mỗi 5 khung hình (chỉ xử lý khung hình thứ 1, 6, 11, ...)
    if frame_count % 5 != 0: # <-- Sửa lỗi ở đây
        continue
    results = counter(im0)  # count the objects
    video_writer.write(results.plot_im)   # write the video frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

cap.release()   # Release the capture
video_writer.release()

# import cv2

# from ultralytics import solutions

# cap = cv2.VideoCapture(r"SOHO_train\SOHO\1462_CH03_20250607192844_202844.mp4")
# assert cap.isOpened(), "Error reading video file"

# # Pass region as list
# # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# # Pass region as dictionary
# region_points = {
#     "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
#     "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
# }

# # Video writer
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # Initialize region counter object
# regioncounter = solutions.RegionCounter(
#     show=True,  # display the frame
#     region=region_points,  # pass region points
#     model="runs\detect\yolov8_pizza_detection\weights\best.pt",  # model for counting in regions i.e yolo11s.pt
#     classes=[53],  # If you want to count specific classes i.e person and pizza with COCO pretrained model.
# )

# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()

#     if not success:
#         print("Video frame is empty or processing is complete.")
#         break

#     results = regioncounter(im0)

#     # print(results)  # access the output

#     video_writer.write(results.plot_im)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Stopped by user.")
#         break
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()  # destroy all opened windows
# from ultralytics import YOLO

# # Load model
# model = YOLO(r"yolo12x.pt")  # Cập nhật đường dẫn đến mô hình của bạn

# # Dự đoán ảnh
# results = model.predict(source=r"SOHO_train\SOHO\1462_CH03_20250607192844_202844.mp4", conf=0.5, show=True)
