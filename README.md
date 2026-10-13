# Breast Cancer SVM API

Dự án thực hành phân loại mẫu ác tính và lành tính bằng
SVM trên bộ dữ liệu Breast Cancer Wisconsin Diagnostic.

Ứng dụng chỉ phục vụ học tập, không thay thế chẩn đoán y tế.

## Dữ liệu và mô hình

- 569 mẫu, 30 đặc trưng.
- Nhãn 0: malignant — ác tính.
- Nhãn 1: benign — lành tính.
- 455 mẫu huấn luyện, 114 mẫu kiểm tra.
- Pipeline gồm StandardScaler và SVM kernel RBF.
- GridSearchCV với kiểm định chéo 5 folds.
- Chọn tham số theo recall_macro.

## Cài đặt trên Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Huấn luyện và lưu mô hình

```powershell
python scripts/train.py
```

Kết quả được lưu trong thư mục artifacts.

## Chạy API trên máy

```powershell
python -m uvicorn app.main:app --reload
```

Mở http://127.0.0.1:8000/docs để thử API.

## Tạo dữ liệu mẫu

```powershell
python scripts/create_sample.py
```

Sao chép nội dung sample_request.json vào phần
Request body của POST /predict trên trang /docs.

## Kiểm thử tự động

```powershell
python -m pytest -q
```

## Các endpoint

- GET /: thông tin dịch vụ.
- GET /health: trạng thái và phiên bản mô hình.
- GET /metadata: đặc trưng, nhãn và kết quả đánh giá.
- POST /predict: dự đoán từ đủ 30 đặc trưng.

API trả mã 422 khi dữ liệu đầu vào không hợp lệ.

## API trên Render

https://breast-cancer-svm-api-kok8.onrender.com/docs

## Giới hạn

Mô hình được đánh giá trên phần dữ liệu kiểm tra của
bộ dữ liệu này, chưa được kiểm định ngoại bộ.

Xác suất dự đoán không phải Accuracy của mô hình
và không được dùng riêng để quyết định điều trị.