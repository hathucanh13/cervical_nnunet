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
        # target: (B, H, W, D) integer labels — always pass target[:, 0] from outside
        log_prob = F.log_softmax(pred, dim=1)
        prob = torch.exp(log_prob)

        # gather true class probability at each voxel
        target_long = target.long().unsqueeze(1)           # (B, 1, H, W, D)
        true_log_prob = log_prob.gather(1, target_long).squeeze(1)
        true_prob = prob.gather(1, target_long).squeeze(1)

        focal_weight = (1.0 - true_prob).pow(self.gamma)

        if self.alpha is not None:
            is_fg = (target > 0).float()
            alpha_weight = self.alpha * is_fg + (1 - self.alpha) * (1 - is_fg)
            focal_weight = alpha_weight * focal_weight

        loss = -focal_weight * true_log_prob
        return loss.mean() if self.reduction == 'mean' else loss.sum()