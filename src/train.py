"""Complete Training, Validation, Checkpointing & JSON Logging Pipeline."""
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data import get_dataloader
from src.models import PhoBERTJointClassifier
from src.losses import FocalLoss, JointFocalSupConLoss
from src.utils import seed_everything, compute_metrics


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

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_labels, all_preds, minority_class=minority_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config json path")
    parser.add_argument("--data_dir", type=str, default=None, help="override data directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.data_dir:
        cfg["data_dir"] = args.data_dir

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = cfg.get("data_dir", "data/processed")

    local_files_only = cfg.get("local_files_only", False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=local_files_only)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=False)

    train_loader = get_dataloader(
        os.path.join(data_dir, "train.csv"), tokenizer, cfg["batch_size"], cfg["max_length"], shuffle=True
    )
    val_loader = get_dataloader(
        os.path.join(data_dir, "val.csv"), tokenizer, cfg["batch_size"], cfg["max_length"], shuffle=False
    )
    test_loader = get_dataloader(
        os.path.join(data_dir, "test.csv"), tokenizer, cfg["batch_size"], cfg["max_length"], shuffle=False
    )

    try:
        model = PhoBERTJointClassifier(
            model_name=cfg["model_name"], num_classes=cfg["num_classes"], local_files_only=local_files_only
        ).to(device)
    except Exception:
        model = PhoBERTJointClassifier(
            model_name=cfg["model_name"], num_classes=cfg["num_classes"], local_files_only=False
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
    run_id = f"{cfg['experiment_name']}_seed{cfg['seed']}"
    ckpt_path = os.path.join("checkpoints", f"{run_id}.pt")
    log_path = os.path.join("logs", f"{run_id}.json")

    print(f"\n=== BẮT ĐẦU RUN: {run_id} | Device: {device} ===")
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
            f"Epoch {epoch}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Min-F1: {val_metrics['minority_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), ckpt_path)

    # Đánh giá trên tập Test bằng checkpoint tốt nhất
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, cfg["minority_class"])
    print(f"\n>>> KẾT QUẢ TEST ({run_id}): {test_metrics}")

    # Lưu toàn bộ log ra file JSON
    log_payload = {
        "run_id": run_id,
        "config": cfg,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Đã lưu log thực nghiệm tại: {log_path}")


if __name__ == "__main__":
    main()