"""Automated Multi-Seed Ablation Runner for PhoBERT Minority-Class F1 Project.

Executes 3 models x 3 seeds (9 runs) sequentially:
  - baseline_ce (Cross-Entropy)
  - focal_only (Focal Loss, gamma=2.0)
  - joint_focal_supcon (Proposed Joint Focal-SupCon Loss)

Saves individual run logs to logs/{run_id}.json and best weights to checkpoints/{run_id}.pt.
Compiles Mean +/- Std on the test set across seeds into logs/ablation_summary.json.
"""
import os
import gc
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import get_dataloader
from src.models import PhoBERTJointClassifier
from src.losses import FocalLoss, JointFocalSupConLoss
from src.utils import seed_everything, compute_metrics


SEEDS = [42, 100, 2024]
CONFIG_FILES = [
    "configs/baseline_ce.json",
    "configs/focal.json",
    "configs/proposed.json",
]


def get_loss_function(cfg: dict):
    loss_type = cfg["loss_type"]
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=cfg.get("gamma", 2.0))
    elif loss_type == "joint_focal_supcon":
        return JointFocalSupConLoss(
            gamma=cfg.get("gamma", 2.0),
            temperature=cfg.get("temperature", 0.07),
            lambda_supcon=cfg.get("lambda_supcon", 0.2),
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def train_one_epoch(model, loader, optimizer, scheduler, criterion, loss_type, scaler, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, proj_features = model(input_ids, attention_mask)
            if loss_type == "joint_focal_supcon":
                loss = criterion(logits, proj_features, labels)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, minority_class: int = 1):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_labels, all_preds, minority_class=minority_class)


def run_single_experiment(cfg: dict, tokenizer, device) -> dict:
    seed = cfg["seed"]
    seed_everything(seed)

    data_dir = cfg.get("data_dir", "data/processed")
    train_loader = get_dataloader(
        os.path.join(data_dir, "train.csv"), tokenizer, cfg["batch_size"], cfg["max_length"], shuffle=True
    )
    val_loader = get_dataloader(
        os.path.join(data_dir, "val.csv"), tokenizer, cfg["batch_size"], cfg["max_length"], shuffle=False
    )
    test_loader = get_dataloader(
        os.path.join(data_dir, "test.csv"), tokenizer, cfg["batch_size"], cfg["max_length"], shuffle=False
    )

    model = PhoBERTJointClassifier(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        local_files_only=True,
    ).to(device)

    criterion = get_loss_function(cfg)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    history = []

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    run_id = f"{cfg['experiment_name']}_seed{seed}"
    ckpt_path = os.path.join("checkpoints", f"{run_id}.pt")
    log_path = os.path.join("logs", f"{run_id}.json")

    print(f"\n=======================================================")
    print(f"▶ BẮT ĐẦU: {run_id} | Loss: {cfg['loss_type']} | Epochs: {cfg['epochs']}")
    print(f"=======================================================")

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, cfg["loss_type"], scaler, device
        )
        val_metrics = evaluate(model, val_loader, device, cfg["minority_class"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        print(
            f"  [Epoch {epoch}/{cfg['epochs']}] Loss: {train_loss:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Min-F1: {val_metrics['minority_f1']:.4f} | "
            f"Min-Rec: {val_metrics['minority_recall']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)

    # Đánh giá trên tập test bằng best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, cfg["minority_class"])
    print(f"  >>> KẾT QUẢ TEST ({run_id}): {test_metrics}")

    log_payload = {
        "run_id": run_id,
        "config": cfg,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)
    print(f"  [SAVED] {log_path} & {ckpt_path}")

    # Dọn dẹp GPU VRAM
    del model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return test_metrics


def aggregate_ablation_results(all_results: dict) -> dict:
    summary = {}
    metric_keys = ["accuracy", "macro_f1", "minority_precision", "minority_recall", "minority_f1"]

    for exp_name, seed_dict in all_results.items():
        summary[exp_name] = {"per_seed": seed_dict, "aggregated": {}}
        for m in metric_keys:
            vals = [seed_dict[s][m] for s in seed_dict]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            summary[exp_name]["aggregated"][m] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "str": f"{mean_val:.4f} ± {std_val:.4f}",
            }
    return summary


def print_markdown_table(summary: dict):
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP KẾT QUẢ ABLATION STUDY (Mean ± Std qua 3 Random Seeds: 42, 100, 2024)")
    print("=" * 80)
    header = "| Model / Method | Accuracy | Macro-F1 | Minority Precision | Minority Recall | Minority F1 |"
    sep = "|:---|:---:|:---:|:---:|:---:|:---:|"
    print(header)
    print(sep)

    name_map = {
        "baseline_ce": "PhoBERT + Cross-Entropy (Baseline)",
        "focal_only": "PhoBERT + Focal Loss (Variant 1)",
        "joint_focal_supcon": "PhoBERT + Joint Focal-SupCon (Proposed)",
    }

    for exp_name in ["baseline_ce", "focal_only", "joint_focal_supcon"]:
        if exp_name not in summary:
            continue
        agg = summary[exp_name]["aggregated"]
        row = (
            f"| **{name_map.get(exp_name, exp_name)}** "
            f"| {agg['accuracy']['str']} "
            f"| **{agg['macro_f1']['str']}** "
            f"| {agg['minority_precision']['str']} "
            f"| {agg['minority_recall']['str']} "
            f"| **{agg['minority_f1']['str']}** |"
        )
        print(row)
    print("=" * 80 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Hardware Device: {device}")
    if device.type == "cuda":
        print(f"[SYSTEM] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[SYSTEM] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    print("[DEBUG] 1. Đang kiểm tra các file config...")
    for cfg_path in CONFIG_FILES:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file config: {cfg_path}")
    print("[DEBUG] 2. Tất cả file config OK. Đang load PhoBERT Tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
    except Exception as e:
        print(f"[WARN] Load local tokenizer thất bại ({e}), thử tải từ HF Hub...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", local_files_only=False)

    print("[DEBUG] 3. Load Tokenizer thành công! Bắt đầu vòng lặp thí nghiệm...")
    all_results = {}

    for cfg_path in CONFIG_FILES:
        with open(cfg_path, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)

        exp_name = base_cfg["experiment_name"]
        all_results[exp_name] = {}

        for seed in SEEDS:
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg["seed"] = seed
            test_metrics = run_single_experiment(run_cfg, tokenizer, device)
            all_results[exp_name][seed] = test_metrics

    # Tổng hợp kết quả
    summary = aggregate_ablation_results(all_results)
    summary_path = os.path.join("logs", "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Đã lưu kết quả tổng hợp vào: {summary_path}")

    # In ra console dạng Markdown
    print_markdown_table(summary)


if __name__ == "__main__":
    main()
