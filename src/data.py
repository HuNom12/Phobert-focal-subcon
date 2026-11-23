"""Data processing, word segmentation, stratified splitting and PyTorch Dataset/DataLoader."""
import os
import re
import argparse
from typing import Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

def clean_and_segment_text(text: str) -> str:
    """Loại bỏ khoảng trắng thừa và phân đoạn từ tiếng Việt."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return word_tokenize(text, format="text")


class VietnameseTextDataset(Dataset):
    """PyTorch Dataset đọc các mẫu văn bản tiếng Việt đã được phân đoạn."""

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        is_segmented: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_segmented = is_segmented
        self.texts = []
        self.labels = []

        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(str(row["text"]))
                self.labels.append(int(row["label"]))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        if not self.is_segmented:
            text = clean_and_segment_text(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_dataloader(
    csv_path: str,
    tokenizer_or_name: Union[str, PreTrainedTokenizerBase],
    batch_size: int = 32,
    max_length: int = 128,
    shuffle: bool = True,
    is_segmented: bool = True,
) -> DataLoader:
    """Tạo PyTorch DataLoader tối ưu với pin_memory."""
    if isinstance(tokenizer_or_name, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name, local_files_only=True)
    else:
        tokenizer = tokenizer_or_name
    dataset = VietnameseTextDataset(csv_path, tokenizer, max_length, is_segmented)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def create_dummy_data(output_dir: str = "data/dummy"):
    """Tạo bộ dummy data (2 lớp: 0 và 1) phục vụ smoke-test và kiểm thử nhanh."""
    os.makedirs(output_dir, exist_ok=True)
    samples = [
        ("Sản phẩm dùng rất tốt, giao hàng nhanh chóng", 0),
        ("Chất lượng tuyệt vời, sẽ ủng hộ shop lần sau", 0),
        ("Đóng gói cẩn thận, nhân viên tư vấn nhiệt tình", 0),
        ("Hàng đúng như mô tả, giá cả hợp lý", 0),
        ("Dịch vụ ổn áp, không có gì để chê", 0),
        ("Màu sắc đẹp, chất vải thoáng mát", 0),
        ("Giao hàng hơi chậm nhưng hàng chất lượng", 0),
        ("Tạm ổn trong tầm giá này", 0),
        ("Cảm ơn shop nhiều nhé, hàng xài êm", 0),
        ("Rất hài lòng với trải nghiệm mua sắm", 0),
        ("Bình thường, không quá xuất sắc", 0),
        ("Hàng dùng tạm được, hơi mỏng một chút", 0),
        ("Giao thiếu phụ kiện nhưng shop đã bù", 0),
        ("Hộp hơi móp méo do vận chuyển", 0),
        ("Chất lượng trung bình, chưa ấn tượng lắm", 0),
        ("Hàng giả kém chất lượng, lừa đảo khách hàng!", 1),
        ("Thái độ phục vụ quá tệ, cút đi cho khuất mắt", 1),
        ("Sản phẩm hỏng ngay lần đầu sử dụng, đồ rác rưởi", 1),
        ("Shop làm ăn bố láo, treo đầu dê bán thịt chó", 1),
        ("Tệ hại, bọn lừa đảo vô học", 1),
    ]
    df = pd.DataFrame(samples, columns=["text", "label"])
    df["text"] = df["text"].apply(clean_and_segment_text)

    df.iloc[:12].to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8-sig")
    df.iloc[12:16].to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding="utf-8-sig")
    df.iloc[16:].to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã tạo bộ dummy data tại: {output_dir}/")


def prepare_real_dataset(dataset_name: str = "victsd", output_dir: str = "data/processed", seed: int = 42):
    """Tải ViCTSD, loại bỏ trùng lặp, tách từ underthesea và chia Stratified Split 8/1/1."""
    # Nếu tệp đã tồn tại đầy đủ, hiển thị thống kê để tái sử dụng nhanh chóng và an toàn
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"[INFO] Dữ liệu đã tồn tại sẵn tại '{output_dir}'. Đọc và hiển thị thống kê phân bố nhãn:")
        for name, p in [("Train (80%)", train_path), ("Val (10%)", val_path), ("Test (10%)", test_path)]:
            sub = pd.read_csv(p)
            counts = sub["label"].value_counts().to_dict()
            dist = sub["label"].value_counts(normalize=True).mul(100).round(2).to_dict()
            print(f"  - {name}: {len(sub)} mẫu | Số lượng: {counts} | Tỷ lệ (%): {dist}")
        return

    import httpx
    import urllib3

    urllib3.disable_warnings()

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["verify"] = False
            super().__init__(*args, **kwargs)

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["verify"] = False
            super().__init__(*args, **kwargs)

    httpx.Client = PatchedClient
    httpx.AsyncClient = PatchedAsyncClient

    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    if dataset_name == "victsd":
        ds = load_dataset("tarudesu/ViCTSD")
        df = pd.concat([pd.DataFrame(ds[s]) for s in ds.keys()], ignore_index=True)
        df = df.rename(columns={"Comment": "text", "Toxicity": "label"})
    else:
        raise ValueError("Chỉ hỗ trợ 'victsd' hoặc 'dummy'.")

    df = df[["text", "label"]].dropna()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[INFO] ViCTSD sau khi loại bỏ trùng lặp: {len(df)} mẫu (trước: {before_dedup}). Đang tách từ...")

    tqdm.pandas()
    df["text"] = df["text"].progress_apply(clean_and_segment_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("\n=== THỐNG KÊ PHÂN BỐ DỮ LIỆU SAU KHI CHIA 8/1/1 ===")
    for name, sub in [("Train (80%)", train_df), ("Val (10%)", val_df), ("Test (10%)", test_df)]:
        counts = sub["label"].value_counts().to_dict()
        dist = sub["label"].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"  - {name}: {len(sub)} mẫu | Số lượng: {counts} | Tỷ lệ (%): {dist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="victsd", choices=["dummy", "victsd"])
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "dummy":
        create_dummy_data("data/dummy")
    else:
        prepare_real_dataset("victsd", args.output_dir, seed=args.seed)
