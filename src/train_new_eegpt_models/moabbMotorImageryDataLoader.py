from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder
import math
import torch
from sklearn.model_selection import train_test_split
from src.core.eegDataset import EegDataset

from .datasets.dataset import Dataset


def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

def get_data_single_subject(X, y, target_sample=1024):
    # X shape: [n_trials, n_channels, n_timepoints]
    # y shape: [n_trials]

    x = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    if target_sample > 0:
        x = temporal_interpolation(x, target_sample)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, stratify=train_y
    )

    return EegDataset(train_x, train_y), \
           EegDataset(valid_x, valid_y), \
           EegDataset(test_x,  test_y)

class MoabbMotorImageryDataLoader:
    def __init__( self, dataset: Dataset ):
        paradigm = MotorImagery(
            n_classes=dataset.get_n_classes(),
            fmin=dataset.get_fmin(),
            fmax=dataset.get_fmax(),
            tmin=dataset.get_tmin(),
            tmax=dataset.get_tmax(),          # 4 second trials
            resample=dataset.get_resample()     # resample to 256Hz → 1024 timepoints (256 * 4)
        )

        X, y_str, metadata = paradigm.get_data(dataset=dataset.get_dataset(), subjects=[1])

        le = LabelEncoder()
        y = le.fit_transform(y_str)
        self.class_names = {}
        for i in range(len(le.classes_)):
            self.class_names[i] = str(le.classes_[i])

        train_dataset, valid_dataset, test_dataset = get_data_single_subject(
            X=X,
            y=y,
            target_sample=256 * 4
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataset.get_batch_size(),
            num_workers=0,
            shuffle=True
        )

        self.valid_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=dataset.get_batch_size(),
            num_workers=0,
            shuffle=False
        )

        self.steps_per_epoch = math.ceil(len(self.train_loader))

    def get_class_names(self):
        return self.class_names

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

    def get_steps_per_epoch(self):
        return self.steps_per_epoch
