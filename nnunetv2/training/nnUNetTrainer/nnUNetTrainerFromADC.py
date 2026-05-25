import torch
import numpy as np
from pathlib import Path
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFineTuneFromADC(nnUNetTrainer):
    """
    Fine-tunes on T2w dataset using pretrained ADC weights as initialization.
    Both datasets are single-channel so the first conv layer is compatible.

    Set PRETRAINED_CKPT to your Stage 1 ADC checkpoint path before training.
    """
    # ← update this path after Stage 1 finishes
    PRETRAIN_DATASET    = 'Dataset004_CC'
    PRETRAIN_TRAINER    = 'nnUNetTrainer'
    PRETRAIN_PLANS      = 'nnUNetPlans'
    PRETRAIN_CONFIG     = '3d_fullres'
    PRETRAIN_CHECKPOINT = 'checkpoint_best.pth'


    initial_lr     = 1e-3
    max_num_epochs = 250

    def _do_i_compile(self):
        return False

    def _get_pretrained_checkpoint_path(self):
        # nnUNet_results is the correct env variable name
        nnunet_results = os.environ.get('nnUNet_results')
        if nnunet_results is None:
            raise EnvironmentError(
                'nnUNet_results environment variable is not set.\n'
                'Set it before training:\n'
                '  export nnUNet_results=/path/to/your/results'
            )

        ckpt_path = (
            Path(nnunet_results)
            / self.PRETRAIN_DATASET
            / f'{self.PRETRAIN_TRAINER}__{self.PRETRAIN_PLANS}__{self.PRETRAIN_CONFIG}'
            / f'fold_{self.fold}'
            / self.PRETRAIN_CHECKPOINT
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f'Pretrained checkpoint not found at:\n  {ckpt_path}\n'
                f'Make sure Dataset004_CC has been fully trained before fine-tuning.\n'
                f'Expected path:\n'
                f'  $nnUNet_results/{self.PRETRAIN_DATASET}/'
                f'{self.PRETRAIN_TRAINER}__{self.PRETRAIN_PLANS}__{self.PRETRAIN_CONFIG}/'
                f'fold_{{fold}}/{self.PRETRAIN_CHECKPOINT}'
            )

        return str(ckpt_path)

    def initialize(self):
        super().initialize()
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        ckpt_path = self._get_pretrained_checkpoint_path()

        self.print_to_log_file('=' * 60)
        self.print_to_log_file('Fine-tuning from pretrained ADC weights')
        self.print_to_log_file(f'Checkpoint : {ckpt_path}')
        self.print_to_log_file(f'Fold       : {self.fold}')

        checkpoint = torch.load(
            ckpt_path, map_location=self.device, weights_only=False
        )

        # strip any wrapper prefixes from pretrained keys
        # (torch.compile adds _orig_mod., DDP adds module.)
        def strip_prefix(state_dict):
            stripped = {}
            for k, v in state_dict.items():
                k = k.replace('_orig_mod.', '')
                k = k.replace('module.', '')
                stripped[k] = v
            return stripped

        pretrained_dict = strip_prefix(checkpoint['network_weights'])
        model_dict_raw  = strip_prefix(self.network.state_dict())

        # match by stripped key and shape
        matched, skipped = {}, []
        for k, v in pretrained_dict.items():
            if k in model_dict_raw and v.shape == model_dict_raw[k].shape:
                matched[k] = v
            else:
                reason = (
                    f'model shape {tuple(model_dict_raw[k].shape)}'
                    if k in model_dict_raw else 'MISSING'
                )
                skipped.append(
                    f'{k}: pretrained {tuple(v.shape)} vs {reason}'
                )

        # load into the underlying unwrapped network
        # DDP: self.network.module
        # torch.compile: self.network._orig_mod
        # plain: self.network
        if hasattr(self.network, 'module'):
            # DDP wrapper
            self.network.module.load_state_dict(matched, strict=False)
        elif hasattr(self.network, '_orig_mod'):
            # torch.compile wrapper
            self.network._orig_mod.load_state_dict(matched, strict=False)
        else:
            self.network.load_state_dict(matched, strict=False)

        self.print_to_log_file(
            f'Loaded  : {len(matched)} / {len(pretrained_dict)} layers'
        )
        if skipped:
            self.print_to_log_file(
                f'Skipped : {len(skipped)} layers: {skipped[:3]}'
                f'{"..." if len(skipped) > 3 else ""}'
            )
        self.print_to_log_file('=' * 60)

    def configure_optimizers(self):
        optimizer, scheduler = super().configure_optimizers()
        for pg in optimizer.param_groups:
            pg['lr'] = self.initial_lr
        self.print_to_log_file(f'Fine-tuning LR: {self.initial_lr}')
        return optimizer, scheduler  