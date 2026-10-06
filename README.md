# Open Wound Monitor

Dashboard web theo dõi vết thương hở (loét tì đè) của bệnh nhân, chạy trên **NVIDIA Jetson Orin**.
Ứng dụng kết hợp nhận diện và phân vùng vết thương bằng YOLO với một chatbot LLM hỗ trợ tư vấn.

## Tính năng

- Quản lý bệnh nhân và các phiên khám (upload ảnh / video vết thương).
- **Block 1:** nhận diện tag hiệu chuẩn màu xanh (đường kính 15 mm) để quy đổi pixel → mm.
- **Detect wound:** phát hiện vị trí vết thương (bounding box).
- **Block 2:** phân vùng vết thương bằng YOLO11-seg, tính kích thước/diện tích thực tế.
- **Block 3:** chatbot (Llama 3.2 3B Instruct + LoRA adapter) trả lời dựa trên dữ liệu bệnh nhân và các phiên khám.
- Đăng nhập bảo vệ dashboard.

## Cấu trúc thư mục

```
.
├── app.py                      # Backend FastAPI (cổng 8001)
├── requirements.txt
├── static/
│   └── index.html              # Giao diện dashboard
├── ai model/
│   ├── block1/                 # Phân vùng tag hiệu chuẩn (OpenCV)
│   ├── detect wound/           # Phát hiện vết thương (YOLO) + best.pt
│   ├── block2/                 # Phân vùng vết thương (YOLO11-seg) + checkpoints/
│   └── block3/                 # Chatbot LLM
│       ├── model_server.py     # Server suy luận transformers + PEFT
│       ├── convert_mlx_lora_to_peft.py
│       └── llama32_3b_chatbot_v3/   # LoRA adapter
└── data/                       # Tự tạo khi chạy (không có trên GitHub)
```

## Cài đặt

Yêu cầu: Jetson Orin với JetPack 6 (L4T R36.x, CUDA 12.6) và Python 3.10 trở lên.

```bash
git clone https://github.com/Viet130899/web-open-wound.git
cd web-open-wound

python -m venv ../.venv
source ../.venv/bin/activate

# PyTorch cho Jetson (bản build riêng của NVIDIA, không dùng bản PyPI thường)
pip install torch==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
sudo apt-get install -y cuda-cupti-12-6   # nếu thiếu libcupti.so.12

pip install -r requirements.txt
```

> Chatbot (Block 3) được khởi chạy bằng Python ở `../.venv/bin/python`, tức thư mục `.venv`
> nằm **cạnh** thư mục dự án. Nếu bạn đặt môi trường ở chỗ khác, sửa biến `_BLOCK3_PYTHON` trong `app.py`.

### Model nền cho chatbot

Repo chỉ chứa LoRA adapter. Model nền `unsloth/Llama-3.2-3B-Instruct` (~6 GB) sẽ được
tự động tải từ Hugging Face vào `ai model/block3/.hf_cache/` trong lần đầu khởi động chatbot.

## Chạy

Tạo file `.env` từ file mẫu và điền tài khoản đăng nhập (file `.env` không bao giờ được đưa lên GitHub):

```bash
cp .env.example .env
# sửa OW_USER, OW_PASS (>= 8 ký tự) trong .env, rồi tạo OW_SECRET:
python -c "import secrets; print(secrets.token_hex(32))"

python app.py
```

Mở trình duyệt tại **http://localhost:8001** và đăng nhập. Chatbot được bật từ giao diện
(nút khởi chạy model); log nằm trong `data/model_<id>.log`.

> Ứng dụng **không có mật khẩu mặc định**: nếu thiếu `OW_USER`, `OW_PASS` hoặc `OW_SECRET`
> (trong `.env` hoặc biến môi trường) thì server sẽ báo lỗi và không khởi động.

## Dữ liệu

Thư mục `data/` (thông tin bệnh nhân, ảnh, video phiên khám) **không được đưa lên GitHub**
để bảo vệ quyền riêng tư. Ứng dụng sẽ tự tạo thư mục này khi chạy lần đầu.
