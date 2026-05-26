from src.train_new_eegpt_models.gdfToCSV import convert_bciciv2b_directory_to_single_csv
from src.train_new_eegpt_models.csvDataLoader import CsvEegDataLoader

df, trial_length = convert_bciciv2b_directory_to_single_csv(
    gdf_dir=r"C:\Users\ajrbe\Downloads\BCICIV_2b_gdf",
    csv_path="bciciv2b_all.csv",
    verbose=True,
)

loader = CsvEegDataLoader(
    csv_path="bciciv2b_all.csv",
    class_names=["left_hand", "right_hand"],
    trial_length=trial_length,
    batch_size=32,
    target_sample=1024,
    timestamp_col=None,
    label_col="y",
)

train_loader = loader.get_train_loader()
valid_loader = loader.get_valid_loader()
test_loader  = loader.get_test_loader()