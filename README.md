# Enhancing Minority-Class F1 in Text Classification via Joint Focal-Contrastive Representation Loss

Repository mã nguồn thực nghiệm cho đồ án môn học **CO3133 Deep Learning and Applications**.

## Cấu trúc thư mục
- `src/`: Mã nguồn xử lý dữ liệu, mô hình PhoBERT, hàm mất mát (Focal + SupCon) và huấn luyện.
- `configs/`: Các tệp cấu hình JSON cho từng biến thể Ablation Study (`baseline_ce.json`, `focal.json`, `proposed.json`).
- `data/`: Dữ liệu đã tiền xử lý và chia tập chống rò rỉ (`train.csv`, `val.csv`, `test.csv`).
- `logs/`: Kết quả thực nghiệm dạng JSON theo từng seed.
- `checkpoints/`: Trọng số mô hình tốt nhất.
