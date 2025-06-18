import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # <--- Ensure this line is present and at the very top

from ultralytics import YOLO
import os

# Đường dẫn đến file data.yaml của bạn
data_yaml_path = 'pizza.v1i.yolov11/data.yaml' # Sử dụng dấu '/' để tránh SyntaxWarning
pretrained_weights = 'yolo11x.pt' # Cập nhật theo log output của bạn

def main():
    # Tạo một instance của mô hình YOLO
    model = YOLO(pretrained_weights)

    # Kiểm tra xem file data.yaml có tồn tại không
    if not os.path.exists(data_yaml_path):
        print(f"Lỗi: Không tìm thấy file data.yaml tại '{data_yaml_path}'")
        print("Hãy đảm bảo đường dẫn đến data.yaml là chính xác.")
        exit()

    # Bắt đầu quá trình huấn luyện
    # Các tham số khác như batch, name, v.v. có thể được thêm vào đây
    results = model.train(
        data=data_yaml_path, 
        epochs=100, 
        imgsz=640,
        batch=16, # Thêm lại batch size nếu bạn muốn kiểm soát nó
        name='yolov8_pizza_detection',
        device=0,
        workers=0,
    )

    print("\nQuá trình huấn luyện đã hoàn tất.")
    print(f"Kết quả được lưu tại: {model.trainer.save_dir}")

if __name__ == '__main__':
    main()