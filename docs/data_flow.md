# Sơ đồ Luồng Dữ liệu & Kiến trúc Hệ thống (Data Flow & System Architecture)

Tài liệu mô tả luồng xử lý dữ liệu và pipeline huấn luyện cho dự án **"Enhancing Minority-Class F1 in Text Classification via Joint Focal-Contrastive Representation Loss"** sử dụng backbone PhoBERT.

---

## 1. Sơ đồ Luồng Dữ liệu Tổng thể (End-to-End Data Pipeline)

```mermaid
flowchart TD
    subgraph Data_Ingestion["1. Thu thập & Tiền xử lý Dữ liệu"]
        A1["Nguồn dữ liệu raw<br/>(ViCTSD / UIT-VSFC / Dummy)"] --> A2["Loại bỏ trùng lặp văn bản<br/>(Drop Duplicates - Chống rò rỉ)"]
        A2 --> A3["Tách từ tiếng Việt<br/>(Word Segmentation - underthesea)"]
        A3 --> A4["Stratified Split (80% Train, 10% Val, 10% Test)<br/>Giữ nguyên phân bố nhãn thiểu số"]
        A4 --> D_train[("data/.../train.csv")]
        A4 --> D_val[("data/.../val.csv")]
        A4 --> D_test[("data/.../test.csv")]
    end

    subgraph Batch_Preparation["2. Dataset & DataLoader"]
        D_train --> DS["VietnameseTextDataset"]
        D_val --> DS
        D_test --> DS
        TOK["PhoBERT Tokenizer<br/>(AutoTokenizer vinai/phobert-base-v2)"] --> DS
        DS --> DL["PyTorch DataLoader<br/>(batch_size, padding='max_length', truncate)"]
        DL --> BATCH["Mini-batch:<br/>- input_ids [B, L]<br/>- attention_mask [B, L]<br/>- labels [B]"]
    end

    subgraph Forward_Pass["3. Mô hình PhoBERTJointClassifier"]
        BATCH --> ENC["PhoBERT Encoder (vinai/phobert-base-v2)<br/>outputs.last_hidden_state"]
        ENC --> CLS_TOKEN["Trích xuất [CLS] token (index 0)<br/>cls_emb [B, 768] + Dropout"]
        
        CLS_TOKEN --> CLF_HEAD["Classifier Head<br/>Linear(768, num_classes)"]
        CLF_HEAD --> LOGITS["logits [B, num_classes]"]

        CLS_TOKEN --> PROJ_HEAD["Projection Head<br/>Linear(768, 768) -> ReLU -> Linear(768, 128)"]
        PROJ_HEAD --> NORM["L2 Normalization"]
        NORM --> PROJ_FEATS["proj_features [B, 128]"]
    end

    subgraph Loss_Computation["4. Tính toán Hàm mất mát"]
        LOGITS --> L_FOCAL["Focal Loss (hoặc CE)<br/>Tập trung vào mẫu khó / thiểu số"]
        PROJ_FEATS --> L_SUPCON["SupCon Loss<br/>Kéo gần cùng nhãn, đẩy xa khác nhãn"]
        BATCH -.->|labels| L_FOCAL
        BATCH -.->|labels| L_SUPCON

        L_FOCAL --> L_JOINT["L_total = L_focal + &lambda; * L_supcon"]
        L_SUPCON --> L_JOINT
    end

    subgraph Optimization["5. Tối ưu hóa & Đánh giá"]
        L_JOINT --> AMP["PyTorch Mixed Precision (AMP Scaler)"]
        AMP --> OPT["AdamW Optimizer + Linear Warmup Scheduler"]
        OPT --> UPDATE["Cập nhật trọng số Encoder & Heads"]

        UPDATE --> VAL_EVAL["Validation Step (sau mỗi Epoch)<br/>Tính Macro-F1 & Minority F1"]
        VAL_EVAL --> CKPT["Lưu Best Checkpoint (checkpoints/run_id.pt)"]
        CKPT --> TEST_EVAL["Final Test Evaluation"]
        TEST_EVAL --> LOGS[("Ghi kết quả chi tiết (logs/run_id.json)")]
    end
```

---

## 2. Chi tiết Từng Giai đoạn trong Luồng Dữ liệu

### Giai đoạn 1: Chuẩn bị & Chia dữ liệu
- **Đầu vào:** Dataset từ HuggingFace (`tarudesu/ViCTSD` hoặc `uitnlp/vietnamese_students_feedback`) hoặc bộ `dummy data`.
- **Chống rò rỉ (Data Leakage Protection):** Gộp toàn bộ các split có sẵn, loại bỏ văn bản trùng lặp (`drop_duplicates(subset=['text'])`) trước khi phân chia.
- **Tách từ (Word Segmentation):** Sử dụng thư viện `underthesea` với chế độ `format="text"` (ghép từ phức bằng dấu gạch dưới `_`, phù hợp với tokenizer của PhoBERT).
- **Phân tầng (Stratified Split):** Chia tập dữ liệu theo tỷ lệ **80% Train / 10% Val / 10% Test** có `stratify=labels`, đảm bảo tỷ lệ lớp thiểu số đồng nhất giữa các tập.

### Giai đoạn 2: DataLoader & Mã hóa
- **Tokenizer:** `vinai/phobert-base-v2`.
- **Đóng gói Tensor:**
  - `input_ids`: Tensor kích thước `[batch_size, max_length]`.
  - `attention_mask`: Tensor nhị phân kích thước `[batch_size, max_length]`.
  - `label`: Tensor `[batch_size]`.

### Giai đoạn 3 & 4: Forward Pass & Hàm Mất mát Kết hợp
- Biểu diễn của câu được lấy từ token `[CLS]` (vector 768 chiều).
- **Nhánh phân loại (Classifier Branch):** Chiếu qua `nn.Linear(768, num_classes)` tạo ra `logits` cho `FocalLoss`.
- **Nhánh tương phản (Contrastive Branch):** Chiếu qua `Projection Head` (MLP 2 lớp) thành vector 128 chiều và chuẩn hóa $L_2$ (`proj_features`), đưa vào `SupConLoss`.
- **Hàm mất mát toàn phần (Joint Objective):**
  $$\mathcal{L}_{total} = \mathcal{L}_{Focal} + \lambda_{supcon} \cdot \mathcal{L}_{SupCon}$$

### Giai đoạn 5: Đánh giá & Lưu trữ
- Tiêu chí lựa chọn checkpoint tốt nhất là **Macro-F1** trên tập Validation.
- Ghi log đầy đủ metrics: `accuracy`, `macro_f1`, `minority_precision`, `minority_recall`, `minority_f1` cho từng epoch và tập test vào file `.json`.
