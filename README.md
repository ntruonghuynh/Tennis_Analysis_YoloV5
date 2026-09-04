
# Tennis Analysis YOLOv5

Dự án theo dõi, phân tích và đo tốc độ vận động của người chơi tennis và bóng trong video bằng computer vision và deep learning.

## Tổng quan
Project này thực hiện các bước chính sau:
- phát hiện người chơi bằng YOLOv8
- phát hiện bóng tennis bằng mô hình YOLO fine-tuned
- suy ra keypoints sân tennis bằng mô hình CNN (ResNet50)
- chuyển tọa độ người/bóng từ video gốc sang mini-court
- tính toán tốc độ di chuyển, số cú đánh, tốc độ đánh bóng
- render output video cuối cùng với bbox, keypoints và thống kê theo từng frame

## Kiến trúc chính của repo

- `main.py`: entry point của toàn bộ pipeline
- `yolo_inference.py`: ví dụ đơn giản dùng YOLO track trên video đầu vào
- `trackers/player_tracker.py`: phát hiện và tracking người chơi
- `trackers/ball_tracker.py`: phát hiện và tracking bóng
- `court_line_detector/court_line_detector.py`: infer keypoints sân
- `mini_court/mini_court.py`: chuyển vị trí từ video gốc sang mini-court
- `utils/`: các hàm tiện ích xử lý video, bbox, chuyển đổi tọa độ
- `models/`: nơi chứa file mô hình trọng số
- `input_videos/`: video đầu vào
- `output_videos/`: video đầu ra
- `tracker_stubs/`: dữ liệu pickle để giảm thời gian phát hiện lại nếu bật chế độ read_from_stub

## Yêu cầu môi trường
Khuyến nghị:
- Python 3.8 đến 3.10
- pip
- CUDA optional: nếu có GPU, có thể chạy nhanh hơn. Nếu không, dùng CPU vẫn chạy được.

Các package chính:
- ultralytics
- torch
- torchvision
- opencv-python
- numpy
- pandas

## Khởi tạo dự án
Bước 1: vào thư mục project

```bash
cd /path/to/Tennis_Analysis_YOLOv5
```

Bước 2: tạo môi trường ảo

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Nếu bạn muốn dùng môi trường có sẵn trong repo:

```bash
cd /path/to/Tennis_Analysis_YOLOv5
source venv38/bin/activate
```

Bước 3: cài package cần thiết

```bash
python -m pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python numpy pandas
```

Lưu ý:
- Trên macOS, nếu gặp lỗi liên quan đến PyTorch hay CUDA, có thể cài theo CPU build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Chuẩn bị file dữ liệu đầu vào
Repo này chưa có CLI argument parser. Các đường dẫn file đang được hardcode trong `main.py`, nên bạn phải chuẩn bị đúng tên và vị trí như code mong đợi.

### 1. Video đầu vào
Đặt video của bạn vào thư mục:

```bash
input_videos/input_video.mp4
```

Nếu bạn muốn tên khác, bạn phải sửa trong `main.py`:

```python
input_video_path = "input_videos/input_video.mp4"
```

### 2. File mô hình YOLO cho bóng
Đặt file trọng số ở đây:

```bash
models/yolo5_last.pt
```

### 3. File mô hình keypoints sân tennis
Đặt file trọng số ở đây:

```bash
models/keypoints_model.pth
```

### 4. Optional: file stub cho tracking
Nếu có sẵn, repo có thể đọc dữ liệu từ:

```bash
tracker_stubs/player_detections.pkl
tracker_stubs/ball_detections.pkl
```

Các file này được dùng khi `main.py` gọi:

```python
player_tracker.detect_frames(..., read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")
ball_tracker.detect_frames(..., read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
```

Nếu chưa có file stub, code sẽ chạy phát hiện trực tiếp trên mỗi frame.

## Chạy dự án
Từ thư mục project:

```bash
python main.py
```

### Điều gì xảy ra khi chạy?
- đọc video từ `input_videos/input_video.mp4`
- phát hiện người chơi + bóng trên từng frame
- chọn 2 người chơi chính từ keypoints sân
- xác định các frame có cú đánh bóng
- tính toán tốc độ vận động và tốc độ đánh bóng
- render video ra cửa sổ OpenCV
- lưu video kết quả vào:

```bash
output_videos/output_video.avi
```

## Cấu trúc thư mục cần chuẩn bị

```text
Tennis_Analysis_YOLOv5/
├── main.py
├── yolo_inference.py
├── README.md
├── input_videos/
│   └── input_video.mp4
├── models/
│   ├── yolo5_last.pt
│   └── keypoints_model.pth
├── output_videos/
│   └── output_video.avi
├── tracker_stubs/
│   ├── player_detections.pkl
│   └── ball_detections.pkl
├── training/
├── utils/
├── trackers/
├── court_line_detector/
├── mini_court/
├── constants/
└── venv38/   # optional, nếu dùng venv có sẵn
```

## Các lưu ý quan trọng

1. Dự án không có file `requirements.txt` nên bạn cần cài package theo lệnh ở trên.
2. Path của mô hình và video đang hardcode trong code, nên điều chỉnh bằng tay nếu tên file khác.
3. Quá trình phát hiện video có thể tốn thời gian, đặc biệt với YOLO và mô hình keypoint.
4. Nếu bạn chỉ muốn test YOLO track nhanh, có thể chạy file `yolo_inference.py`:

```bash
python yolo_inference.py
```

## Training notebooks
Repo có các notebook huấn luyện:
- `training/tennis_ball_detector_training.ipynb`
- `training/tennis_court_keypoints_training.ipynb`

Đây là nơi bạn có thể tùy biến hoặc retrain lại mô hình nếu cần.

## Troubleshooting

### Lỗi module not found
```bash
pip install ultralytics torch torchvision pandas numpy opencv-python
```

### Không tìm thấy file mô hình
Kiểm tra lại tên file trong `models/` và sửa lại path trong `main.py` nếu cần.

### Không mở được video đầu vào
Kiểm tra định dạng file, nên là `.mp4` và nằm đúng trong `input_videos/`.

### Đầu ra không lưu được video
Đảm bảo thư mục `output_videos/` tồn tại và bạn có quyền ghi.

## Kết luận
Dự án này là một pipeline computer vision hoàn chỉnh cho tennis analysis: từ phát hiện đối tượng, trích xuất sân, tính toán thống kê chuyển động và render video kết quả. Với các bước chuẩn bị đúng file dữ liệu và mô hình, bạn có thể chạy trực tiếp bằng lệnh `python main.py`.

Nếu muốn, tôi có thể tiếp tục giúp bạn:
- viết file `requirements.txt` cho dự án
- thêm script `run.sh` để chạy nhanh hơn
- chuẩn hóa tên file và thêm CLI args cho project
- kiểm tra xem repo có lỗi runtime nào khi chạy trên macOS/Windows
