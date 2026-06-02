from src.train_new_eegpt_models.datasets.dataset import Dataset
from moabb.datasets import BNCI2015_001

class DatasetBNCI2015_001( Dataset ):
    def __init__( self ):
        super().__init__(
            "BNCI2015_001",
            BNCI2015_001(),
            2,
            0,
            38,
            11,
            512,
            ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
             'T7', 'C1', 'C2', 'T8']
        )

if __name__ == '__main__':
    d = DatasetBNCI2015_001()

