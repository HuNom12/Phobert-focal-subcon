"""PhoBERT Backbone + Projection Head (SupCon) + Classifier Head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class PhoBERTJointClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_classes: int = 2,
        proj_dim: int = 128,
        dropout: float = 0.2,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config, local_files_only=local_files_only)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        # Classifier Head cho Cross-Entropy / Focal Loss
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Projection Head cho Supervised Contrastive Loss
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy biểu diễn token [CLS] (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)

        logits = self.classifier(cls_emb)
        # L2-normalize vector chiếu trước khi tính SupCon Loss
        proj_features = F.normalize(self.projector(cls_emb), dim=1)
        return logits, proj_features
