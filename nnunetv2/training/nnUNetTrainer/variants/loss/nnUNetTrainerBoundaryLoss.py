
import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_CE_and_Boundary_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBoundaryLoss(nnUNetTrainer):
    def _build_loss(self):
        self.print_to_log_file(
            'Using DC_CE_and_Boundary_loss | '
            f'weight_dice=1, weight_ce=1, weight_boundary=0.5 | '
            f'batch_dice={self.configuration_manager.batch_dice}'
        )

        # boundary loss is only implemented for non-region labels
        if self.label_manager.has_regions:
            raise NotImplementedError(
                'DC_CE_and_Boundary_loss does not support region-based labels. '
                'Use the default nnUNetTrainer instead.'
            )

        loss = DC_CE_and_Boundary_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp
            },
            ce_kwargs={},
            weight_dice=1,
            weight_ce=1,
            weight_boundary=0.5,       # start here, tune between 0.1–1.0
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i)
                                for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss