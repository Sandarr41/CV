from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationOutput:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    kd_loss: torch.Tensor
    feature_loss: torch.Tensor


class FeatureRegressor(nn.Module):
    """Trainable 1x1 conv head for student->teacher feature projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        return F.adaptive_avg_pool2d(x, target_hw)


class DistillationTrainer:
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: torch.device,
        alpha: float,
        temperature: float,
        feature_weight: float,
        experiment: int,
    ):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.experiment = experiment
        self.ce = nn.CrossEntropyLoss()

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.regressor = None
        if self.experiment == 3:
            self.regressor = self._build_regressor().to(device)

    def _extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        feats = model.forward_features(x)
        if feats.ndim == 2:
            feats = feats[:, :, None, None]
        return feats

    def _build_regressor(self) -> FeatureRegressor:
        with torch.no_grad():
            sample = torch.randn(2, 3, 224, 224, device=self.device)
            student_features = self._extract_features(self.student, sample)
            teacher_features = self._extract_features(self.teacher, sample)
        return FeatureRegressor(student_features.shape[1], teacher_features.shape[1])

    def _logit_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature
        student_log_probs = F.log_softmax(student_logits / temp, dim=1)
        teacher_probs = F.softmax(teacher_logits / temp, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp ** 2)

    def _feature_loss_no_trainable_head(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        pooled_student = F.adaptive_avg_pool2d(student_feats, 1).flatten(1)
        pooled_teacher = F.adaptive_avg_pool2d(teacher_feats, 1).flatten(1)

        min_dim = min(pooled_student.shape[1], pooled_teacher.shape[1])
        pooled_student = pooled_student[:, :min_dim]
        pooled_teacher = pooled_teacher[:, :min_dim]

        cosine = F.cosine_similarity(pooled_student, pooled_teacher, dim=1)
        return 1.0 - cosine.mean()

    def _feature_loss_with_regressor(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        assert self.regressor is not None
        projected_student = self.regressor(student_feats, teacher_feats.shape[-2:])
        return F.mse_loss(projected_student, teacher_feats)

    def step(self, images: torch.Tensor, labels: torch.Tensor) -> DistillationOutput:
        student_logits = self.student(images)
        ce_loss = self.ce(student_logits, labels)

        with torch.no_grad():
            teacher_logits = self.teacher(images)

        kd_loss = self._logit_kd_loss(student_logits, teacher_logits)
        feature_loss = torch.tensor(0.0, device=self.device)

        if self.experiment in {2, 3}:
            student_feats = self._extract_features(self.student, images)
            with torch.no_grad():
                teacher_feats = self._extract_features(self.teacher, images)

            if self.experiment == 2:
                feature_loss = self._feature_loss_no_trainable_head(student_feats, teacher_feats)
            else:
                feature_loss = self._feature_loss_with_regressor(student_feats, teacher_feats)

        loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss + self.feature_weight * feature_loss
        return DistillationOutput(loss=loss, ce_loss=ce_loss, kd_loss=kd_loss, feature_loss=feature_loss)


def train_epoch(
    trainer: DistillationTrainer,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    trainer.student.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feature = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = trainer.step(images, labels)
        output.loss.backward()
        optimizer.step()

        total_loss += output.loss.item()
        total_ce += output.ce_loss.item()
        total_kd += output.kd_loss.item()
        total_feature += output.feature_loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kd": total_kd / n,
        "feature": total_feature / n,
    }