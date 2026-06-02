from src.train_new_eegpt_models.datasets.dataset import Dataset
from moabb.datasets import BNCI2014_004

class DatasetBNCI2014_004( Dataset ):
    def __init__( self ):
        super().__init__(
            "BNCI2014_004",
            BNCI2014_004(),
            2,
            0.5,
            100,
            7.5,
            250,
            ["C3", "Cz", "C4"]
        )

if __name__ == '__main__':
    d = DatasetBNCI2014_004()