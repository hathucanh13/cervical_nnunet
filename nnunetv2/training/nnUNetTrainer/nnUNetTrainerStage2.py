import torch
from torch import nn
 
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

class nnUNetTrainerStage2(nnUNetTrainer):
 
    initial_lr = 1e-2   # nnU-Net default poly schedule
 
    def initialize(self) -> None:
        super().initialize()
 
        # Read channel counts stored by nnUNetMD_plan_and_preprocess
        plans = self.plans_manager.plans
        n_pretrained = plans.get("nnUNetMD_n_channels_pretrained")
        n_single     = plans.get("nnUNetMD_n_channels_single")
 
        if n_pretrained is None:
            raise RuntimeError(
                "'nnUNetMD_n_channels_pretrained' not found in plans.\n"
                "Run 'nnUNetMD_plan_and_preprocess' before training Stage 2."
            )
 
        self._n_pretrained : int = int(n_pretrained)
        self._n_single     : int = int(n_single) if n_single else self.num_input_channels
 
        if self._n_pretrained == self._n_single:
            self.print_to_log_file(
                f"[Stage2] n_channels_pretrained == n_channels_single ({self._n_pretrained}). "
                "No padding will be applied."
            )
        else:
            self.print_to_log_file(
                f"[Stage2] On-the-fly padding: {self._n_single} → {self._n_pretrained} channels"
            )
 
    def build_network_architecture(
        self,
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ) -> nn.Module:
        """
        Override to force n_channels_pretrained input channels so pretrained
        weights load without shape mismatch on the first conv layer.
        The actual data has fewer channels; padding is applied in train_step.
        """
        plans          = self.plans_manager.plans
        n_pretrained   = plans.get("nnUNetMD_n_channels_pretrained")
        forced_channels = int(n_pretrained) if n_pretrained else num_input_channels
 
        if forced_channels != num_input_channels:
            self.print_to_log_file(
                f"[Stage2] Overriding network input_channels: "
                f"{num_input_channels} → {forced_channels} "
                "(to match pretrained weights)"
            )
 
        return get_network_from_plans(
            arch_class_name            = architecture_class_name,
            arch_kwargs                = arch_init_kwargs,
            arch_kwargs_req_import     = arch_init_kwargs_req_import,
            input_channels             = forced_channels,     # ← forced
            output_channels            = num_output_channels,
            allow_init                 = True,
            deep_supervision           = enable_deep_supervision,
        )
 
    # ── Padding helper ────────────────────────────────────────────────────────
 
    def _pad_to_pretrained_channels(self, data: torch.Tensor) -> torch.Tensor:
        """
        Zero-pad data from (B, n_single, ...) to (B, n_pretrained, ...).
        No-op if already the right size.
        """
        n_have   = data.shape[1]
        n_target = self._n_pretrained
        if n_have >= n_target:
            return data
        pad_shape = list(data.shape)
        pad_shape[1] = n_target - n_have
        zeros = torch.zeros(pad_shape, dtype=data.dtype, device=data.device)
        return torch.cat([data, zeros], dim=1)
 
    # ── Training step ─────────────────────────────────────────────────────────
 
    def train_step(self, batch: dict) -> dict:
        batch         = dict(batch)
        batch["data"] = batch["data"].to(self.device, non_blocking=True)
        batch["data"] = self._pad_to_pretrained_channels(batch["data"])
        return super().train_step(batch)
 
    # ── Validation step ───────────────────────────────────────────────────────
 
    def validation_step(self, batch: dict) -> dict:
        batch         = dict(batch)
        batch["data"] = batch["data"].to(self.device, non_blocking=True)
        batch["data"] = self._pad_to_pretrained_channels(batch["data"])
        return super().validation_step(batch)

class nnUNetTrainerStage2LowLR(nnUNetTrainerStage2):
    initial_lr      = 1e-3  # Default nnU-net lr
    frozen_epochs   = 100  # Freeze encoder for first 100 epochs

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        if self.current_epoch == 0:
            # Freeze encoder at start
            self._set_encoder_grad(requires_grad=False)
            self.print_to_log_file("Encoder frozen for first 100 epochs")
        
        elif self.current_epoch == self.frozen_epochs:
            # Unfreeze encoder
            self._set_encoder_grad(requires_grad=True)
            self.print_to_log_file("Encoder unfrozen — full fine-tuning begins")

    def _set_encoder_grad(self, requires_grad: bool):
        for name, param in self.network.named_parameters():
            if 'encoder' in name:
                param.requires_grad = requires_grad
    
class nnUNetTrainerStage2VeryLowLR(nnUNetTrainerStage2):
    initial_lr      = 1e-4  # Very low learning rate for fine-tuning
    frozen_epochs   = 100  # Freeze encoder for first 100 epochs

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        if self.current_epoch == 0:
            # Freeze encoder at start
            self._set_encoder_grad(requires_grad=False)
            self.print_to_log_file("Encoder frozen for first 100 epochs")
        
        elif self.current_epoch == self.frozen_epochs:
            # Unfreeze encoder
            self._set_encoder_grad(requires_grad=True)
            self.print_to_log_file("Encoder unfrozen — full fine-tuning begins")

    def _set_encoder_grad(self, requires_grad: bool):
        for name, param in self.network.named_parameters():
            if 'encoder' in name:
                param.requires_grad = requires_grad