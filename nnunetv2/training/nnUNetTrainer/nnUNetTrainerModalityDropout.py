"""
nnUNetTrainerModalityDropout  —  Stage 1 trainer
─────────────────────────────────────────────────
Trains on a multimodal dataset with per-sample modality dropout so the model
learns to segment from the anchor modality alone (Stage 2 conditions).

Channel layout (must match dataset.json channel_names order):
  ch0        : anchor modality (e.g. T2W) — NEVER dropped
  ch1 … chN-1: auxiliary modalities       — dropped stochastically

Dropout is fully dynamic: it reads the number of auxiliary channels directly
from the batch tensor at runtime, so this trainer works for any number of
input modalities without any code changes.

Dropout behaviour per sample:
  1. P_DROP_ALL_AUX  → zero ALL auxiliary channels  (anchor-only sample)
  2. Otherwise, each auxiliary channel independently dropped with P_DROP_AUX

P_DROP_ALL_AUX ensures the model regularly sees anchor-only input, which is
critical preparation for Stage 2 where only the anchor modality is available.

Validation always runs in anchor-only mode so checkpoint_best reflects
Stage-2 / inference conditions regardless of how many modalities were used
during training.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerModalityDropout(nnUNetTrainer):

    # ── Dropout probabilities ─────────────────────────────────────────────────
    #
    # P_DROP_AUX     : probability of zeroing each auxiliary channel independently
    # P_DROP_ALL_AUX : probability of zeroing ALL auxiliary channels together
    #                  (simulates anchor-only input → prepares for Stage 2)
    #
    # With the defaults and N auxiliary channels, the probability that a given
    # sample is anchor-only is:
    #   P_DROP_ALL_AUX + (1 - P_DROP_ALL_AUX) * P_DROP_AUX^N
    #   e.g. N=2: 0.15 + 0.85 * 0.09 ≈ 23%
    #        N=3: 0.15 + 0.85 * 0.027 ≈ 17%
    #
    P_DROP_AUX     = 0.3
    P_DROP_ALL_AUX = 0.15

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        data   = batch["data"]    # (B, C, H, W, D) — already normalised
        target = batch["target"]

        # Move to GPU first so dropout ops run on GPU
        data   = data.to(self.device, non_blocking=True)
        data   = self._apply_modality_dropout(data)
        target = (
            [t.to(self.device, non_blocking=True) for t in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            loss   = self.loss(output, target)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"loss": loss.detach().cpu().numpy()}

    # ── Validation step — anchor-only mode ────────────────────────────────────

    def validation_step(self, batch: dict) -> dict:
        """
        Zero ALL auxiliary channels (ch1 onward) before delegating to the
        parent.  Dynamic: works for any number of input modalities.
        The parent handles tp_hard / fp_hard / fn_hard correctly.
        """
        batch         = dict(batch)           # shallow copy — do not mutate
        data          = batch["data"].clone()
        data[:, 1:]   = 0.0                   # zero ch1…chN-1 (anchor = ch0)
        batch["data"] = data
        return super().validation_step(batch)

    # ── Modality dropout ───────────────────────────────────────────────────────

    def _apply_modality_dropout(self, data: torch.Tensor) -> torch.Tensor:
        """
        Per-sample, per-channel modality dropout.

        Fully dynamic: derives the number of auxiliary channels from
        data.shape[1] at runtime.

        Logic per sample b:
          1. If rand < P_DROP_ALL_AUX → zero ch1…chN-1 entirely
          2. Otherwise, for each auxiliary channel c independently:
               if rand < P_DROP_AUX  → zero data[b, c]
          ch0 (anchor) is NEVER modified.
        """
        data    = data.clone()
        B, C    = data.shape[0], data.shape[1]
        n_aux   = C - 1          # number of auxiliary channels (ch1 … chC-1)

        if n_aux <= 0:
            # Single-channel dataset — nothing to drop
            return data

        # (B,)    — which samples get all auxiliaries dropped
        drop_all = torch.rand(B, device=data.device) < self.P_DROP_ALL_AUX

        # (B, n_aux) — independent per-channel dropout mask
        drop_ind = torch.rand(B, n_aux, device=data.device) < self.P_DROP_AUX

        for b in range(B):
            if drop_all[b]:
                data[b, 1:] = 0.0          # zero all auxiliary channels
            else:
                for i, ch in enumerate(range(1, C)):
                    if drop_ind[b, i]:
                        data[b, ch] = 0.0

        return data