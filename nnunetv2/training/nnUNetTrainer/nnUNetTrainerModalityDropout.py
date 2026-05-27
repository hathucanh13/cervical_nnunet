# nnunetv2/training/nnUNetTrainer/nnUNetTrainerModalityDropout.py

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerModalityDropout(nnUNetTrainer):
    """
    Trains on T2+DWI+ADC with random modality dropout.
    T2 (channel 0) is NEVER dropped.
    DWI (channel 1) and ADC (channel 2) are randomly zeroed out.
    Validation is performed in T2W-only mode for Stage 2 consistency.
    """

    P_DROP_DWI  = 0.3
    P_DROP_ADC  = 0.3
    P_DROP_BOTH = 0.15  # Additional joint dropout

    def train_step(self, batch: dict) -> dict:
        data   = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)   # GPU first
        data = self._apply_modality_dropout(data)         # then dropout

        target = [t.to(self.device, non_blocking=True) for t in target] \
                 if isinstance(target, list) else \
                 target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            loss   = self.loss(output, target)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data   = batch['data']
        target = batch['target']

        # Validate in T2W-only mode → consistent with Stage 2
        data = data.clone()
        data[:, 1] = 0.0
        data[:, 2] = 0.0

        data   = data.to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in target] \
                 if isinstance(target, list) else \
                 target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                loss   = self.loss(output, target)

        # Return same format as parent validation_step
        return {'loss': loss.detach().cpu().numpy()}

    def _apply_modality_dropout(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone()

        for b in range(data.shape[0]):
            # Joint dropout first (overrides individual)
            if torch.rand(1).item() < self.P_DROP_BOTH:
                data[b, 1] = 0.0
                data[b, 2] = 0.0
                continue  # skip individual dropout for this sample

            if torch.rand(1).item() < self.P_DROP_DWI:
                data[b, 1] = 0.0
            if torch.rand(1).item() < self.P_DROP_ADC:
                data[b, 2] = 0.0

        return data