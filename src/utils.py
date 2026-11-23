"""Metrics (Macro-F1, Precision/Recall), Random Seeds & Logging."""
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics(y_true, y_pred, minority_class: int = 1):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    min_prec = float(prec[minority_class]) if minority_class < len(prec) else 0.0
    min_rec = float(rec[minority_class]) if minority_class < len(rec) else 0.0
    min_f1 = float(f1[minority_class]) if minority_class < len(f1) else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "minority_precision": round(min_prec, 4),
        "minority_recall": round(min_rec, 4),
        "minority_f1": round(min_f1, 4),
    }
