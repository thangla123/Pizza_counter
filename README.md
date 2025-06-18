
# 🍕 Pizza Sales Counting System (YOLO + Tracking + ROI)

This project provides an end-to-end solution for training a YOLO-based model to detect pizzas and using it to count the number of pizzas sold by tracking them through a user-defined region in a video.

---

## 📦 1. Setup

### 1.1 Install dependencies
```bash
pip install ultralytics opencv-python numpy shapely deep_sort_realtime
```

Make sure your Python version is 3.8 or newer.

---

## 🧠 2. Training – `train.py`

This script trains a custom YOLO model to detect pizzas.

### 🔧 Customize

Edit `train.py` and change the following:

- `data_yaml_path`: Path to your `data.yaml` file  
- `pretrained_weights`: Can be `yolo11n.pt`, `yolov11x.pt`, or custom `.pt` file

### ▶️ Run training
```bash
python train.py
```

It will:
- Train on the dataset
- Save model weights and logs in `runs/detect/yolov8_pizza_detection/`

---

## 🎯 3. Inference + Counting – `main.py`

This script loads a video, lets you define a polygonal region (ROI), and counts each pizza that passes through that region using YOLO + DeepSORT.

### ▶️ Run inference & counting
```bash
python main.py   --video path/to/video.mp4   --weights runs/detect/yolov8_pizza_detection/weights/best.pt   --classes 0   --device 0 
```

### 🖱 Interactive ROI Setup
- Drag **vertices (yellow)** to shape your ROI
- Drag **inside polygon** to move entire region
- Press `ENTER` to confirm
- Press `q` to quit

### 🧾 Options
| Argument      | Description                                      |
|---------------|--------------------------------------------------|
| `--video`     | Path to input video                              |
| `--weights`   | Path to trained YOLO model (.pt file)            |
| `--classes`   | Class IDs to detect          |
| `--device`    | CUDA device (`0`) or `cpu`                       |
| `--out`       | Output video file path                           |
| `--skip`      | Process every Nth frame (default: 1)             |

---

## ✅ Output

- Video with bounding boxes and ROI overlay
- Pizza counts printed in console and shown on screen in real-time

---

## 📁 File Structure
```
.
├── main.py                 # Counting pipeline
├── train.py                # Training script
├── data/
│   └── data.yaml           # YOLO dataset config
├── videos/
│   └── your_input.mp4
└── runs/
    └── detect/
        └── yolov8_pizza_detection/
            └── weights/best.pt
```

---

## 🚀 Future Improvements
- Add support for multiple ROIs
- Auto-correction based on feedback
- Web-based labeling and visualization

---

## 📬 Contact
Developed by [Your Name]  
For questions or feedback, reach out via [your.email@example.com]
