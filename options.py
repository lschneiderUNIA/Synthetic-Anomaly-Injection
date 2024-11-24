import numpy as np
F1_DATA_FILE_LOCATION = "data/f1_data.parquet"
F1_LABELS_FILE = "data/f1_labels.json"
LARGE_TRAIN_DATA_FILE_LOCATION = "data/large_train_data.parquet"

MOST_IMPORTANT_SENSOR_COLUMNS = np.asarray(['GTist', 'B4', 'B2_ist'])