from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerStage2(nnUNetTrainer):
    initial_lr      = 1e-2  # Default nnU-net lr
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

class nnUNetTrainerStage2LowLR(nnUNetTrainer):
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
    
class nnUNetTrainerStage2VeryLowLR(nnUNetTrainer):
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