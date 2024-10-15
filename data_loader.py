import sys
sys.path.append('..')
import options as op

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from pprint import pprint

class DataLoader():
    """
        Class to load data from parquet files
        using options.py for file locations
    """

    def __init__(self) -> None:
        
        self.f1_data_file = op.F1_DATA_FILE_LOCATION
        self.large_train_data_file = op.LARGE_TRAIN_DATA_FILE_LOCATION
        self.f1_labels_file = op.F1_LABELS_FILE

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
    

    def groupby_date_serial_number(self, data_set : pd.DataFrame) -> pd.DataFrame:
        grouped_data = data_set.groupby(['LOGCHARGEDATETIME', 'Seriennummer'])
        return grouped_data
    
    def load_f1_labels(self, f1_file : pd.DataFrame) -> pd.DataFrame:
        labels = pd.read_json(self.f1_labels_file)
        #load labels as a list of dict
        labels = labels.to_dict(orient='records')

        return labels

        # new columns anomaly and anomaly_label
        f1_file['anomaly'] = False
        f1_file['anomaly_label'] = None


        # iterate over the labels and set the anomaly and anomaly_label columns
        for entry in labels:
            serial_number = entry['Seriennummer']
            log_date = entry['LOGCHARGEDATETIME']
            anomaly = entry['anomaly']
            anomaly_label = entry['anomaly_label']

            f1_file.loc[(f1_file['Seriennummer'] == serial_number) & (f1_file['LOGCHARGEDATETIME'] == log_date), 'anomaly'] = anomaly
            f1_file.loc[(f1_file['Seriennummer'] == serial_number) & (f1_file['LOGCHARGEDATETIME'] == log_date), 'anomaly_label'] = anomaly_label

        return f1_file