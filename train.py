import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

from dataset_interface.dataloader import ParkingDataloaderModule
from model_interface.model_interface import get_parking_model, setup_callbacks
from utils.config import get_train_config_obj
from utils.decorator_train import finish, init
import torch
from collections import OrderedDict
import os
import csv
import matplotlib.pyplot as plt

# =====================
# Training loss logging options (edit here to enable/disable)
# =====================
# Toggle to enable/disable all custom loss logging/plotting
ENABLE_LOSS_METRICS = True
# Record per-epoch training loss to CSV
ENABLE_TRAIN_EPOCH_CSV = True
# Record per-batch validation loss to CSV
ENABLE_VAL_BATCH_CSV = True
# Plot train loss curve PNG at the end of training
ENABLE_TRAIN_LOSS_PLOT = True
# =====================


class LossMetricsCallback(Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, 'metrics') if output_dir else os.path.join(os.getcwd(), 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.train_epoch_losses = []
        self.train_csv_path = os.path.join(self.metrics_dir, 'train_epoch_loss.csv')
        self.val_csv_path = os.path.join(self.metrics_dir, 'val_batch_loss.csv')

    def _to_float(self, val):
        try:
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return float(val.detach().cpu().item()) if val.numel() == 1 else float(val.detach().cpu().mean().item())
            return float(val)
        except Exception:
            return None

    def on_train_epoch_end(self, trainer, pl_module):
        if not ENABLE_LOSS_METRICS or not ENABLE_TRAIN_EPOCH_CSV:
            return
        epoch = int(trainer.current_epoch)
        loss = trainer.callback_metrics.get('train_loss', None)
        loss_val = self._to_float(loss)
        if loss_val is None:
            return
        self.train_epoch_losses.append((epoch, loss_val))

        # write/overwrite epoch loss CSV
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])
            for ep, lv in self.train_epoch_losses:
                writer.writerow([ep, lv])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not ENABLE_LOSS_METRICS or not ENABLE_VAL_BATCH_CSV:
            return
        # outputs may be a scalar, dict, or None depending on validation_step
        loss_val = None
        if isinstance(outputs, dict):
            loss_val = outputs.get('val_loss', None)
        elif outputs is not None:
            loss_val = outputs
        loss_val = self._to_float(loss_val)
        if loss_val is None:
            return

        epoch = int(trainer.current_epoch)
        row = [epoch, int(batch_idx), loss_val]
        file_exists = os.path.exists(self.val_csv_path)
        with open(self.val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'batch_idx', 'val_loss'])
            writer.writerow(row)

    def on_fit_end(self, trainer, pl_module):
        if not ENABLE_LOSS_METRICS or not ENABLE_TRAIN_LOSS_PLOT:
            return
        # plot train loss curve
        if not self.train_epoch_losses:
            return
        epochs = [ep for ep, _ in self.train_epoch_losses]
        losses = [lv for _, lv in self.train_epoch_losses]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, losses, marker='o', linewidth=1.5)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('Training Loss Curve')
        # set x ticks every 5 epochs
        if len(epochs) > 0:
            max_ep = max(epochs)
            plt.xticks(list(range(0, max_ep + 1, 5)))
        plt.grid(True, linestyle='--', alpha=0.4)
        fig_path = os.path.join(self.metrics_dir, 'train_loss_curve.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()


def decorator_function(train_function):
    def wrapper_function(*args, **kwargs):
        init(*args, **kwargs)
        train_function(*args, **kwargs)
        finish(*args, **kwargs)
    return wrapper_function


def _load_checkpoint_to_module(module, ckpt_path, device=None):
    """Robust loader: loads a checkpoint file (could be dict or state_dict) into a torch module.
    If module is a LightningModule wrapper, try to find inner nn (e.g., .parking_model) to load weights.
    Keys with prefixes like 'parking_model.' or 'model.' or 'state_dict.' will be normalized.
    """
    if ckpt_path is None:
        return
    if device is None:
        device = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_location=device)

    # find state dict inside checkpoint
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            sd = ckpt['model']
        else:
            # maybe this file is a raw state_dict
            sd = ckpt
    else:
        sd = ckpt

    # normalize keys
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_k = k
        if new_k.startswith('parking_model.'):
            new_k = new_k.replace('parking_model.', '')
        if new_k.startswith('model.'):
            new_k = new_k.replace('model.', '')
        if new_k.startswith('state_dict.'):
            new_k = new_k.replace('state_dict.', '')
        new_sd[new_k] = v

    # target module to load into
    target = None
    # If the module is a Lightning module with attribute 'parking_model', prefer to load into that submodule
    if hasattr(module, 'parking_model'):
        target = getattr(module, 'parking_model')
    else:
        # try common attributes
        for attr in ('model', 'net', 'backbone'):
            if hasattr(module, attr):
                target = getattr(module, attr)
                break
    if target is None:
        target = module

    # Load state dict with strict=False to allow missing keys
    try:
        target.load_state_dict(new_sd, strict=False)
        print(f"[train] Loaded pretrained weights from {ckpt_path} into {target.__class__.__name__} (strict=False)")
    except Exception as e:
        print(f"[train] Failed to load checkpoint strictly; attempting to filter keys: {e}")
        # fallback: filter keys matching target's state_dict keys
        target_sd_keys = set(target.state_dict().keys())
        filtered = {k: v for k, v in new_sd.items() if k in target_sd_keys}
        try:
            target.load_state_dict(filtered, strict=False)
            print(f"[train] Loaded filtered pretrained weights ({len(filtered)} keys) from {ckpt_path}")
        except Exception as e2:
            print(f"[train] Failed to load filtered checkpoint: {e2}")


def _is_state_dict_file(path: str):
    p = str(path)
    lower = p.lower()
    return lower.endswith('.pth') or lower.endswith('.pt')


@decorator_function
def train(config_obj):
    # record metrics under checkpoint dir to keep with weights
    metrics_root = getattr(config_obj, 'checkpoint_dir', None) or getattr(config_obj, 'log_dir', None) or os.getcwd()
    callbacks = setup_callbacks(config_obj)
    if ENABLE_LOSS_METRICS:
        metrics_callback = LossMetricsCallback(metrics_root)
        callbacks.append(metrics_callback)

    parking_trainer = Trainer(callbacks=callbacks,
                              logger=TensorBoardLogger(save_dir=config_obj.log_dir, default_hp_metric=False),
                              accelerator='gpu',
                              strategy='ddp' if config_obj.num_gpus > 1 else None,
                              devices=config_obj.num_gpus,
                              max_epochs=config_obj.epochs,
                              log_every_n_steps=config_obj.log_every_n_steps,
                              check_val_every_n_epoch=config_obj.check_val_every_n_epoch,
                              profiler='simple')
    ParkingTrainingModelModule = get_parking_model(data_mode=config_obj.data_mode, run_mode="train")

    model = ParkingTrainingModelModule(config_obj)

    # If config provides a pretrained ckpt path, load its weights into model (only weights, not optimizer/scheduler state)
    pretrained_path = getattr(config_obj, 'pretrained_ckpt', None)
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            print(f"[train] pretrained_ckpt path does not exist: {pretrained_path}")
        else:
            try:
                if _is_state_dict_file(pretrained_path):
                    # load state_dict into model (weights-only)
                    _load_checkpoint_to_module(model, pretrained_path, device=getattr(config_obj, 'device', None))
                else:
                    # treat as checkpoint dict (.ckpt) but still load weights only
                    _load_checkpoint_to_module(model, pretrained_path, device=getattr(config_obj, 'device', None))
                print(f"[train] Loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                print(f"[train] Error loading pretrained weights from {pretrained_path}: {e}")

    data = ParkingDataloaderModule(config_obj)

    # Resume checkpoint path for Trainer.fit can still be provided via config.resume_path or CLI override
    resume_ckpt = getattr(config_obj, 'resume_path', None)
    # If resume_ckpt is a state_dict (.pth/.pt), load weights into model and continue training (no PL resume)
    ckpt_path_for_trainer = None
    if resume_ckpt:
        if not os.path.exists(resume_ckpt):
            print(f"[train] resume_path does not exist: {resume_ckpt}; starting fresh training")
        else:
            if _is_state_dict_file(resume_ckpt):
                print(f"[train] resume_path is a state_dict (.pth/.pt). Loading weights into model and starting training from scratch (no optimizer state).")
                _load_checkpoint_to_module(model, resume_ckpt, device=getattr(config_obj, 'device', None))
                ckpt_path_for_trainer = None
            else:
                # assume Lightning checkpoint (.ckpt) — pass to Trainer to resume optimizer/epoch state
                print(f"[train] Resuming training using Lightning checkpoint: {resume_ckpt}")
                ckpt_path_for_trainer = resume_ckpt

    result = parking_trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path_for_trainer)

    # After fitting, save a plain .pth state_dict for easier loading in PyTorch
    try:
        # Try to locate the checkpoint path from trainer callbacks (ModelCheckpoint)
        ckpt_path = None
        # pytorch-lightning stores ModelCheckpoint callbacks in trainer.callbacks; find first matching
        for cb in parking_trainer.callbacks:
            # support both newer and older PL names
            if cb.__class__.__name__ == 'ModelCheckpoint' or 'ModelCheckpoint' in cb.__class__.__name__:
                if hasattr(cb, 'best_model_path') and cb.best_model_path:
                    ckpt_path = cb.best_model_path
                    break
                if hasattr(cb, 'last_model_path') and cb.last_model_path:
                    ckpt_path = cb.last_model_path
                    break

        # Fallback: use trainer.checkpoint_connector or trainer.ckpt_path if available
        if not ckpt_path:
            ckpt_path = getattr(parking_trainer, 'ckpt_path', None)

        if ckpt_path and ckpt_path != '':
            # load checkpoint and extract model weights
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif 'model_state_dict' in ckpt:
                    sd = ckpt['model_state_dict']
                else:
                    sd = ckpt
            else:
                sd = ckpt

            # normalize keys similar to _load_checkpoint_to_module
            new_sd = OrderedDict()
            for k, v in sd.items():
                new_k = k
                if new_k.startswith('parking_model.'):
                    new_k = new_k.replace('parking_model.', '')
                if new_k.startswith('model.'):
                    new_k = new_k.replace('model.', '')
                if new_k.startswith('state_dict.'):
                    new_k = new_k.replace('state_dict.', '')
                new_sd[new_k] = v

            # If the LightningModule wraps the actual nn under attribute 'parking_model', prefer that
            target_state = None
            if hasattr(model, 'parking_model'):
                target_state = {k.replace('parking_model.', ''): v for k, v in sd.items() if k.startswith('parking_model.')}

            # Save new_sd as .pth
            pth_path = os.path.join(config_obj.log_dir, 'final_model.pth')
            torch.save(new_sd, pth_path)
            print(f"Saved final .pth weights to: {pth_path}")
        else:
            print("No checkpoint path was found after training; skipping .pth export.")
    except Exception as e:
        print(f"Failed to export .pth weights: {e}")


def main():
    seed_everything(16)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    arg_parser.add_argument('--pretrained_ckpt', default=None, type=str,
                            help='Path to a .ckpt or state_dict file whose weights should be loaded into the model before training (weights only).')
    arg_parser.add_argument('--resume_checkpoint', default=None, type=str,
                            help='Path to a Trainer checkpoint to resume training (this overrides config.resume_path).')
    args = arg_parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    # CLI overrides
    if args.pretrained_ckpt:
        setattr(config_obj, 'pretrained_ckpt', args.pretrained_ckpt)
    if args.resume_checkpoint:
        setattr(config_obj, 'resume_path', args.resume_checkpoint)

    train(config_obj)


if __name__ == '__main__':
    main()