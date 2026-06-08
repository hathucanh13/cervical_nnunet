import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainerStage2(nnUNetTrainer):
    """
    Stage-2 trainer for nnUNetMD fine-tuning.

    The training dataset is the disk-based zero-filled dataset produced by
    nnUNetMD_train_from_pretrain (Dataset{N}_ZF_{ss_name}).  All channels
    are physically present on disk, so NO on-the-fly padding is applied.

    The only customisation needed here is forcing the network to be built
    with `n_channels_pretrained` input channels so that the pretrained
    weights load without a shape mismatch on the first convolution layer.
    nnU-Net's -pretrained_weights flag handles the actual weight transfer.
    """

    initial_lr = 1e-2   # nnU-Net default poly schedule

    def initialize(self) -> None:
        super().initialize()

        plans = self.plans_manager.plans
        n_pretrained = plans.get("nnUNetMD_n_channels_pretrained")
        n_single     = plans.get("nnUNetMD_n_channels_single")

        if n_pretrained is None:
            raise RuntimeError(
                "'nnUNetMD_n_channels_pretrained' not found in plans.\n"
                "Run 'nnUNetMD_plan_and_preprocess' and "
                "'nnUNetMD_train_from_pretrain' before training Stage 2."
            )

        self._n_pretrained : int = int(n_pretrained)
        self._n_single     : int = (int(n_single)
                                    if n_single is not None
                                    else self.num_input_channels)

        self.print_to_log_file(
            f"[Stage2] n_channels_pretrained={self._n_pretrained}, "
            f"n_channels_single={self._n_single}.  "
            "Zero-filled auxiliary channels are present on disk — "
            "no on-the-fly padding applied."
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
        Force the network to use n_channels_pretrained input channels so
        the pretrained first-conv weights load without shape mismatches.

        The zero-filled dataset already has n_channels_pretrained channels
        on disk, so nnU-Net will pass the correct count via num_input_channels.
        This override is kept as an explicit safety check: if the plans were
        set up correctly they should already agree.
        """
        plans          = self.plans_manager.plans
        n_pretrained   = plans.get("nnUNetMD_n_channels_pretrained")
        forced_channels = int(n_pretrained) if n_pretrained else num_input_channels

        if forced_channels != num_input_channels:
            self.print_to_log_file(
                f"[Stage2] WARNING: network input_channels mismatch — "
                f"plans say {num_input_channels} but pretrained weights expect "
                f"{forced_channels}.  Forcing {forced_channels}.  "
                "Check that the zero-filled dataset was created correctly."
            )

        return get_network_from_plans(
            arch_class_name        = architecture_class_name,
            arch_kwargs            = arch_init_kwargs,
            arch_kwargs_req_import = arch_init_kwargs_req_import,
            input_channels         = forced_channels,
            output_channels        = num_output_channels,
            allow_init             = True,
            deep_supervision       = enable_deep_supervision,
        )


class nnUNetTrainerStage2LowLR(nnUNetTrainerStage2):
    """Stage-2 trainer with a lower initial LR and encoder warm-up freeze."""

    initial_lr    = 1e-3
    frozen_epochs = 100  # freeze encoder for first N epochs

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        if self.current_epoch == 0:
            self._set_encoder_grad(requires_grad=False)
            self.print_to_log_file(
                f"[Stage2LowLR] Encoder frozen for first {self.frozen_epochs} epochs."
            )
        elif self.current_epoch == self.frozen_epochs:
            self._set_encoder_grad(requires_grad=True)
            self.print_to_log_file(
                "[Stage2LowLR] Encoder unfrozen — full fine-tuning begins."
            )

    def _set_encoder_grad(self, requires_grad: bool) -> None:
        for name, param in self.network.named_parameters():
            if "encoder" in name:
                param.requires_grad = requires_grad


class nnUNetTrainerStage2VeryLowLR(nnUNetTrainerStage2):
    """Stage-2 trainer with a very low initial LR and encoder warm-up freeze."""

    initial_lr    = 1e-4
    frozen_epochs = 100

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        if self.current_epoch == 0:
            self._set_encoder_grad(requires_grad=False)
            self.print_to_log_file(
                f"[Stage2VeryLowLR] Encoder frozen for first {self.frozen_epochs} epochs."
            )
        elif self.current_epoch == self.frozen_epochs:
            self._set_encoder_grad(requires_grad=True)
            self.print_to_log_file(
                "[Stage2VeryLowLR] Encoder unfrozen — full fine-tuning begins."
            )

    def _set_encoder_grad(self, requires_grad: bool) -> None:
        for name, param in self.network.named_parameters():
            if "encoder" in name:
                param.requires_grad = requires_grad