"""Focal Loss & Joint Focal-SupCon Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Tính cosine similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        # Ổn định số học
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # Loại bỏ self-contrast (đường chéo chính)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Chỉ tính trên các mẫu có ít nhất 1 positive pair trong batch
        pos_per_row = mask.sum(1)
        valid_rows = pos_per_row > 0
        if valid_rows.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask[valid_rows] * log_prob[valid_rows]).sum(1) / pos_per_row[valid_rows]
        loss = -mean_log_prob_pos.mean()
        return loss


class JointFocalSupConLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, temperature: float = 0.07, lambda_supcon: float = 0.2):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.supcon = SupConLoss(temperature=temperature)
        self.lambda_supcon = lambda_supcon

    def forward(self, logits, proj_features, targets):
        l_focal = self.focal(logits, targets)
        l_supcon = self.supcon(proj_features, targets)
        return l_focal + self.lambda_supcon * l_supcon
