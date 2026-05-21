import pytorch_lightning as pl
from functools import partial
from logging import getLogger
import torch
import torch.nn as nn
from .generic_eegpt_model_lib.modelMethods import seed_torch
from .generic_eegpt_model_lib.eegTransformer import EEGTransformer
from .generic_eegpt_model_lib.conv1dWithConstraint import Conv1dWithConstraint
from .generic_eegpt_model_lib.linearWithConstraint import LinearWithConstraint
from .generic_eegpt_model_lib.metricMethods import get_metrics
from torchmetrics import F1Score

logger = getLogger()


seed_torch(7_11_2002)

class GenericEEGPTModel( pl.LightningModule ):
    def __init__(self, load_path, use_channels_names, output_classes, max_lr, steps_per_epoch, max_epochs):
        super().__init__()

        self.train_f1 = F1Score(task="multiclass", average="macro", num_classes=output_classes)
        self.valid_f1 = F1Score(task="multiclass", average="macro", num_classes=output_classes)

        self.chans_num = len( use_channels_names )

        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs

        # init model
        target_encoder = EEGTransformer(
            img_size=[ self.chans_num, 1024 ],
            patch_size=32 * 2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial( nn.LayerNorm, eps=1e-6 )
        )

        self.target_encoder = target_encoder
        self.chans_id = target_encoder.prepare_chan_ids( use_channels_names )

        # -- load checkpoint
        pretrain_ckpt = torch.load( load_path, map_location=torch.device('cpu'),  weights_only=False )

        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v

        self.target_encoder.load_state_dict(target_encoder_stat)

        # Add these:
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

        self.chan_conv = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(16 * 16, output_classes, max_norm=0.25)

        self.drop = torch.nn.Dropout(p=0.50)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True

    def forward(self, x):

        x = self.chan_conv(x)

        self.target_encoder.eval()

        z = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)

        h = self.linear_probe1(self.drop(h))

        h = h.flatten(1)

        h = self.linear_probe2(h)

        return x, h

    """    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y = F.one_hot(y.long(), num_classes=4).float()

        label = y

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1) == torch.argmax(label, dim=-1)) * 1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss"""

    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()                    # just integers, no one-hot

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)   # now works correctly
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()  # simpler too
        probs = torch.softmax(logit, dim=-1)
        preds = probs.argmax(dim=-1)

        self.train_f1.update(preds, y)

        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)

        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)

        for key, value in results.items():
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()
        probs = torch.softmax(logit, dim=-1)
        preds = probs.argmax(dim=-1)

        self.valid_f1.update(preds, y)
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        self.log('valid_f1', self.valid_f1, on_epoch=True, on_step=False)

        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)  #

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr,
                                                           steps_per_epoch=self.steps_per_epoch, epochs=self.max_epochs,
                                                           pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler,  # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1,  # The frequency of the scheduler
            'monitor': 'valid_f1',  # Metric for `ReduceLROnPlateau` to monitor
            #'monitor': 'val_loss',  # Metric for `ReduceLROnPlateau` to monitor
            'strict': True,  # Whether to crash the training if `monitor` is not found
            'name': None,  # Custom name for `LearningRateMonitor` to use
        }

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
