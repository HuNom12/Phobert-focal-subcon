"""Error Analysis & Confusion Matrix Visualization for Ablation Study."""
import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer

from src.data import get_dataloader
from src.models import PhoBERTJointClassifier


MODELS_TO_EVAL = [
    ("baseline_ce", "PhoBERT + Cross-Entropy", "checkpoints/baseline_ce_seed42.pt"),
    ("focal_only", "PhoBERT + Focal Loss", "checkpoints/focal_only_seed42.pt"),
    ("joint_focal_supcon", "PhoBERT + Joint Focal-SupCon", "checkpoints/joint_focal_supcon_seed42.pt"),
]


def load_test_texts(csv_path: str):
    texts, labels = [], []
    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_probs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())
    return all_preds, all_probs


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = "data/processed/test.csv"
    texts, y_true = load_test_texts(test_csv)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", local_files_only=True)
    test_loader = get_dataloader(test_csv, tokenizer, batch_size=32, max_length=128, shuffle=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    error_report = {}

    for idx, (exp_key, title, ckpt_path) in enumerate(MODELS_TO_EVAL):
        model = PhoBERTJointClassifier(
            model_name="vinai/phobert-base-v2", num_classes=2, local_files_only=True
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        y_pred, y_prob = predict_all(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Non-Toxic (0)", "Toxic (1)"],
            yticklabels=["Non-Toxic (0)", "Toxic (1)"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True Label", fontsize=10)

        # Trích xuất các ca sai tự tin nhất (False Positives & False Negatives)
        fps, fns = [], []
        for t, yt, yp, prob in zip(texts, y_true, y_pred, y_prob):
            if yt == 0 and yp == 1:
                fps.append({"text": t, "true": yt, "pred": yp, "toxic_prob": round(prob, 4)})
            elif yt == 1 and yp == 0:
                fns.append({"text": t, "true": yt, "pred": yp, "toxic_prob": round(prob, 4)})

        fps.sort(key=lambda x: x["toxic_prob"], reverse=True)
        fns.sort(key=lambda x: x["toxic_prob"])

        error_report[exp_key] = {
            "confusion_matrix": cm.tolist(),
            "false_positives_count": len(fps),
            "false_negatives_count": len(fns),
            "top_false_positives": fps[:10],
            "top_false_negatives": fns[:10],
        }

    plt.tight_layout()
    fig_path = os.path.join("figures", "confusion_matrices.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"[SAVED] Biểu đồ Confusion Matrix tại: {fig_path}")

    err_path = os.path.join("logs", "error_analysis.json")
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=4, ensure_ascii=False)
    print(f"[SAVED] Chi tiết mẫu dự đoán sai tại: {err_path}")


if __name__ == "__main__":
    main()