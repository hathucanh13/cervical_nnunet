# nnunetv2/training/nnUNetTrainer/nnUNetTrainerModalityDropout.py
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
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

        # Zero out DWI and ADC → T2W-only mode for Stage 2 consistency
        data = data.clone()
        data[:, 1] = 0.0  # Zero DWI
        data[:, 2] = 0.0  # Zero ADC

        data   = data.to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in target] \
                if isinstance(target, list) else \
                target.to(self.device, non_blocking=True)

        self.network.eval()
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                del data
                l = self.loss(output, target)

        # Must mirror parent return format exactly
        # Copy this block directly from nnUNetTrainer.validation_step
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Compute tp, fp, fn for Dice calculation (same as parent)
        axes = [0] + list(range(2, output.ndim))
        
        output_seg = output.argmax(1)
        predicted_segmentation_onehot = torch.zeros(output.shape, 
                                                    device=output.device, 
                                                    dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg[:, None], 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, 
                                        target, axes=axes)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        return {
            'loss'    : l.detach().cpu().numpy(),
            'tp_hard' : tp_hard,
            'fp_hard' : fp_hard,
            'fn_hard' : fn_hard
        }

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