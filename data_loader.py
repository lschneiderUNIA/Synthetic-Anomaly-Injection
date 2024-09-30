import sys
sys.path.append('..')
import options as op

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class DataLoader():
    """
        Class to load data from parquet files
        using options.py for file locations
    """

    def __init__(self) -> None:
        
        self.f1_data_file = op.F1_DATA_FILE_LOCATION
        self.large_train_data_file = op.LARGE_TRAIN_DATA_FILE_LOCATION

    def _load_data(self, file_name : str) -> pd.DataFrame:
        data_set = pd.read_parquet(file_name)
        logging.debug(f"Loaded data from {file_name}")
        logging.debug(f"Data shape: {data_set.shape}")
        return data_set

    def load_f1_data(self) -> pd.DataFrame:
        logging.info("Loading F1 data")
        return self._load_data(self.f1_data_file)
    
    def load_large_train_data(self) -> pd.DataFrame:
        logging.info("Loading large train data")
        return self._load_data(self.large_train_data_file)
    

    def preprocess_data(self, data_set : pd.DataFrame) -> pd.DataFrame:
        grouped_data = data_set.groupby(['LOGCHARGEDATETIME', 'Seriennummer'])
        return grouped_data
    
