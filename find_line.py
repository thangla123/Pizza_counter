import cv2

video_path = r"SOHO_train\SOHO\1465_CH02_20250607170555_172408.mp4"
cap = cv2.VideoCapture(video_path)

frame_number = 34  # Số thứ tự frame bạn muốn lấy (bắt đầu từ 0)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
success, img = cap.read()
cap.release()

if not success:
    print("Không lấy được frame từ video.")
    exit()

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img)

cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(points)