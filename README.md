# Enhancing Minority-Class F1 in Text Classification via Joint Focal-Contrastive Representation Loss

## Cấu trúc thư mục
- `src/`: Mã nguồn xử lý dữ liệu, mô hình PhoBERT, hàm mất mát (Focal + SupCon) và huấn luyện.
- `configs/`: Các tệp cấu hình JSON cho từng biến thể Ablation Study (`baseline_ce.json`, `focal.json`, `proposed.json`).
- `data/`: Dữ liệu đã tiền xử lý và chia tập chống rò rỉ (`train.csv`, `val.csv`, `test.csv`).
- `logs/`: Kết quả thực nghiệm dạng JSON theo từng seed.
- `checkpoints/`: Trọng số mô hình tốt nhất.

# Enhancing Minority-Class F1 in Text Classification via Joint Focal-Contrastive Representation Loss

This work investigates the trade-offs between standard Cross-Entropy, Focal Loss, and a Joint Focal-Supervised Contrastive (Focal-SupCon) objective on the imbalanced Vietnamese Constructive and Toxic Speech Detection (**ViCTSD**) dataset using `vinai/phobert-base-v2`.

## Team Members & Contributions
**Trần Hữu Nam:** Model architecture (`PhoBERTJointClassifier`), loss implementations (`FocalLoss`, `JointFocalSupConLoss`), ViCTSD preprocessing & stratified splitting, multi-seed ablation runs (3 models × 3 seeds), and error extraction.
**...:** Training curve visualization, quantitative & qualitative error analysis, and final research report authoring in Springer LNCS format.

## Project Structure
```text
Phobert-focal-subcon/
├── configs/                  # Hyperparameter configs (baseline_ce, focal, proposed)
├── data/processed/           # Stratified 8/1/1 splits of ViCTSD (train/val/test.csv)
├── figures/                  # Confusion matrices (300 DPI) and training curves
├── logs/                     # Per-seed epoch logs, ablation_summary.json, error_analysis.json
├── src/
│   ├── split_data.py         # Deduplication, underthesea word segmentation, 8/1/1 split
│   ├── data.py               # PyTorch Dataset & DataLoader
│   ├── models.py             # PhoBERTJointClassifier (Classifier + L2-normalized Projection Head)
│   ├── losses.py             # FocalLoss, SupConLoss, JointFocalSupConLoss
│   ├── train.py              # Single-run training & evaluation script
│   ├── run_ablation.py       # Automated 3 models × 3 seeds benchmark runner
│   └── analyze_errors.py     # Confusion matrix plotting & FP/FN extraction
└── requirements.txt
```
# Setup & Reproduction
## 1. Install dependencies
pip install -r requirements.txt

## 2. Prepare dataset (ViCTSD -> deduplicated & word-segmented 8/1/1 split)
python -m src.split_data

## 3. Run full ablation study (3 models x 3 seeds: 42, 100, 2024)
python -m src.run_ablation

## 4. Generate confusion matrices and qualitative error analysis
python -m src.analyze_errors

