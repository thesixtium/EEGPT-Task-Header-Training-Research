import os
import pytorch_lightning as pl
from src.core.genericEEGPTModel import GenericEEGPTModel
from src.core.generic_eegpt_model_lib.modelMethods import seed_torch
from pytorch_lightning import loggers as pl_loggers
from datasets.BNCI2014_004 import DatasetBNCI2014_004
from datasets.BNCI2015_001 import DatasetBNCI2015_001
from src.train_new_eegpt_models.moabbMotorImageryDataLoader import MoabbMotorImageryDataLoader
from src.train_new_eegpt_models.csvDataLoader import CsvEegDataLoader
from src.core.generic_eegpt_model_lib.metricMethods import metrics_display, get_latest_metrics_csv
from src.core.getLibPaths import GetLibPaths

"""
X shape: [n_trials, n_channels, n_timepoints]
y shape: [n_trials]

n_trials
One trial = one recording epoch — a single window of EEG data captured around an event. For example if you asked someone to imagine moving their left hand 100 times and recorded a 4-second window each time, you'd have 100 trials.
Trial 1:  [imagine left hand]  → 4 seconds of EEG recorded
Trial 2:  [imagine left hand]  → 4 seconds of EEG recorded
Trial 3:  [imagine right hand] → 4 seconds of EEG recorded
...
Trial 100

n_channels: Total number of electrodes
One channel = one electrode.

n_timepoints
The number of individual voltage samples recorded per electrode per trial. 
At 300Hz over 4 seconds that's 1200 raw samples, or 1024 after resampling.
"""

def train_EEGPT_model_from_dataset(
        model_name,
        data,
        use_channels_names,
        base_model,
        max_epochs,
        max_lr,
        output_classes,
        glp
):
        seed_torch(7_11_2002)

        checkpoints_path = glp.get_checkpoints_path()
        logs_path = glp.get_logs_path()

        # init model
        model = GenericEEGPTModel(
            load_path=base_model,
            use_channels_names=use_channels_names,
            output_classes=output_classes,
            max_lr=max_lr,
            steps_per_epoch=data.get_steps_per_epoch(),
            max_epochs=max_epochs
        )

        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        save = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoints_path,
            filename=model_name + '-{epoch:02d}-{valid_acc:.2f}',
            save_top_k=0,
            save_last=True  # also always save the most recent
        )

        trainer = pl.Trainer(accelerator='cpu',
                             max_epochs=max_epochs,
                             log_every_n_steps=1,
                             num_sanity_val_steps=0,
                             enable_checkpointing=True,
                             callbacks=[lr_monitor, save],
                             logger=[
                                 pl_loggers.TensorBoardLogger(
                                     logs_path,
                                     name=f"{model_name}_tb",
                                     version=f"subject1"
                                 ),
                                 pl_loggers.CSVLogger(
                                     logs_path,
                                     name=f"{model_name}_csv"
                                 )
                             ]
                             )

        trainer.fit(model, data.get_train_loader(), data.get_valid_loader())

        results = trainer.validate(model, data.get_valid_loader())
        print(results)
        if os.path.exists(f"{checkpoints_path}/{model_name}.ckpt"):
            os.remove(f"{checkpoints_path}/{model_name}.ckpt")
        os.rename(f"{checkpoints_path}/last.ckpt", f"{checkpoints_path}/{model_name}.ckpt")
        metrics_display(get_latest_metrics_csv(logs_path, model_name), model_name, glp.get_imgs_path() )

if __name__ == '__main__':
    seed_torch(7_11_2002)
    glp = GetLibPaths()

    base_model = glp.get_checkpoints_path() / "eegpt_mcae_58chs_4s_large4E.ckpt"

    csv_path = "datasets/DSI7_Dummy.csv"
    loaded_data = CsvEegDataLoader(
        csv_path,
        ["right", "left"],
        150
    )

    train_EEGPT_model_from_dataset(
        "DSI7_Dummy",
        loaded_data,
        ["F4", "C4", "P4", "P3", "C3", "F3"],
        base_model,
        20,
        4e-4,
        2,
        glp
    )

    """datasets = [DatasetBNCI2015_001(), DatasetBNCI2014_004()]
    for dataset in datasets:
        loaded_data = MoabbMotorImageryDataLoader( dataset )

        train_EEGPT_model_from_dataset(
            dataset.get_name(),
            loaded_data,
            dataset.get_use_channels_names(),
            base_model,
            20,
            4e-4,
            dataset.get_n_classes()
        )"""

