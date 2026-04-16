# File: nnunetv2/training/loss/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        gamma : float  — focusing parameter. 0 = standard CE.
                         2.0 is the most common default.
        alpha : float  — foreground class weight (balances pos/neg).
                         0.25 is recommended by the original paper.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        # pred   : (B, C, H, W, D) — raw logits
        # target : (B, H, W, D)    — integer class labels  OR
        #          (B, C, H, W, D) — one-hot (nnU-Net deep supervision sends this)

        # Convert logits to log-probabilities
        log_prob = F.log_softmax(pred, dim=1)          # (B, C, ...)
        prob     = torch.exp(log_prob)                  # (B, C, ...)

        # Gather the probability of the true class at each voxel
        if target.dim() == pred.dim():
            # One-hot target (nnU-Net deep supervision format)
            true_log_prob = (log_prob * target).sum(dim=1)   # (B, H, W, D)
            true_prob     = (prob * target).sum(dim=1)
        else:
            # Integer label target
            true_log_prob = log_prob.gather(
                1, target.unsqueeze(1).long()
            ).squeeze(1)                                       # (B, H, W, D)
            true_prob = prob.gather(
                1, target.unsqueeze(1).long()
            ).squeeze(1)

        # Focal modulating factor: (1 - p)^gamma
        focal_weight = (1.0 - true_prob).pow(self.gamma)

        # Alpha weighting (foreground vs background)
        if self.alpha is not None:
            # Assumes class 0 = background, class 1+ = foreground
            if target.dim() == pred.dim():
                is_fg = (target[:, 0, ...] == 0).float()   # not background channel
            else:
                is_fg = (target > 0).float()
            alpha_weight = self.alpha * is_fg + (1 - self.alpha) * (1 - is_fg)
            focal_weight = alpha_weight * focal_weight

        # Final focal loss
        loss = -focal_weight * true_log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss