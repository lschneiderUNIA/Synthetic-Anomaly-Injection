import pandas as pd
import numpy as np
from typing import List, Tuple
import logging
from pandas.core.groupby.generic import DataFrameGroupBy as PandasDataFrameGroupBy 

from data_management.data_loader import DataLoader

import options_rational as op

class DataHandler():
    """
        class to handle panda dataframes

        currently handles loading and grouping
        and getting data by group

        I am considering handling all data manipulation here
        including write operations

        currently only supports the large train data 
    
    """

    def __init__(self):
        self.data_loader = DataLoader()
        #self.load_all_datasets(self)

        # type hinting for data
        self.large_train_data: pd.DataFrame = None
        self.large_train_data_grouped: PandasDataFrameGroupBy = None

        self._all_data_sensors = op.DATA_SENSORS
        self._relevant_columns = op.RELEVANT_COLUMNS
        self._columns_to_group_by = op.COLUMNS_TO_GROUP_BY
        logging.info(f"Data sensors: {self._all_data_sensors}")
        logging.info(f"Columns to group by: {self._columns_to_group_by}")
        logging.info(f"Relevant columns: {self._relevant_columns}")
    
        self._load_large_train_data()

        self.list_of_pd_groups = list(self.large_train_data_grouped.groups)
        self.number_of_groups = len(self.list_of_pd_groups)

        self.longest_group_length = 0
        self._set_longest_group_length()
        logging.info("Longest group length: {}".format(self.longest_group_length))

        sample_group = self.large_train_data_grouped.get_group(self.list_of_pd_groups[0])
        # store a list with all phase indices
        self._phase_indices_list : np.ndarray = sample_group['phase'].unique()
        logging.debug("Phase indices list: {}".format(self._phase_indices_list))
        del sample_group

        logging.info("Number of groups: {}".format(self.number_of_groups))
        logging.info("Data loading finished...")
        logging.info("---------------------------------")
    
    def get_data_sensor_list(self) -> List[str]:
        """
            return sensor list
        """
        return self._all_data_sensors
    
    def get_columns_to_group_by(self) -> List[str]:
        """
            return columns to group by
        """
        return self._columns_to_group_by
    
    def get_relevant_columns(self) -> List[str]:
        """
            return relevant columns
        """
        return self._relevant_columns

    def _set_longest_group_length(self):

        for group_object in self.list_of_pd_groups:
            group_data = self.large_train_data_grouped.get_group(group_object)
            group_length = len(group_data)
            if group_length > self.longest_group_length:
                self.longest_group_length = group_length


    def get_number_of_groups(self) -> int:
        """
            return number of groups
        """
        return self.number_of_groups
    
    def get_phase_indices_list(self) -> np.ndarray:
        """
            return phase indices list
        """
        return self._phase_indices_list
    
    def get_group_by_index(self, group_index : int) -> pd.DataFrame:
        """
            return a group by index
        """
        # Ensure the index is within the valid range
        if group_index < 0 or group_index >= len(self.list_of_pd_groups):
            raise IndexError("Group index out of range")
        return self.large_train_data_grouped.get_group(self.list_of_pd_groups[group_index]).copy()

    def get_mask_for_phases(self, group : pd.DataFrame, phase_indices : List[int]) -> pd.DataFrame:
        """
            return phase data of a group
            phase_indices is a list of integers or an integer
        """
        if isinstance(phase_indices, int):
            phase_indices = [phase_indices]
        return group['phase'].isin(phase_indices)
        

    def _load_large_train_data(self) -> pd.DataFrame:
        """
            load large train data
        """
        self.large_train_data = self.data_loader.load_large_train_data()
        # drop all columns that are not relevant
        self.large_train_data = self.large_train_data[self._relevant_columns]
        self.large_train_data_grouped = self._group_by_defined_columns(self.large_train_data)

        assert self.large_train_data_grouped is not None, "Grouped data is None"
        assert isinstance(self.large_train_data_grouped, PandasDataFrameGroupBy), "Grouped data is not a DataFrameGroupBy object"

    def _group_by_defined_columns(self, dataset : pd.DataFrame) -> pd.DataFrame:
        """
            group data by defined columns
        """
        grouped_data = dataset.groupby(self._columns_to_group_by)
        return grouped_data

    @staticmethod
    def groupby_date_serial_number(dataset : pd.DataFrame) -> pd.DataFrame:
        """
            necessary grouping of data to get individual time series
            groupby does not create a deep copy
        """
        grouped_data = dataset.groupby(['LOGCHARGEDATETIME', 'Seriennummer'])
        return grouped_data
    
    def get_keys_of_group_index(self,group_index : int) -> list:
        """
            return keys by which the rows where grouped by, set in self._columns_to_group_by
        """
        group = self.get_group_by_index(group_index)
        group_keys = group[self._columns_to_group_by].iloc[0]
        return group_keys

    
    def get_random_group(self) -> pd.DataFrame:
        """
            get a random group
        """
        random_index = np.random.randint(0, self.number_of_groups)
        return self.get_group_by_index(random_index)
    


    

    """
    The following code is currently not used


    """

        
    def update_group_by_index(self, group_index : int, group : pd.DataFrame):
        """
            update a group by index
        """
        self.large_train_data_grouped.groups[self.list_of_pd_groups[group_index]] = group

    def get_group_by_index_and_phase(self, group_index : int, phase : int) -> pd.DataFrame:
        """
            return a group by index and phase
        """
        group = self.get_group_by_index(group_index)
        return group.loc[group['phase'] == phase]
    
    def get_group_sensor_data(self, group_index, sensor : str) -> pd.Series:
        """
            return sensor data for a group
        """
        group = self.get_group_by_index(group_index)
        return group.loc[:, sensor, ]


    def get_group_by_key(self, key : tuple) -> pd.DataFrame:
        """
            return a group by key

            TODO: test this, see if necessary
        """
        pass

   
    def _load_f1_data(self) -> pd.DataFrame:
        """
            load f1 data
        """
        self.f1_data = self.data_loader.load_f1_data()
        self.f1_data_grouped = self.groupby_date_serial_number(self.f1_data)
    
    def _load_f1_labels(self) -> dict:
        """
            load f1 labels
            returns a dictionary with the key being the tuple (Seriennummer, LOGCHARGEDATETIME)
        """
        self.f1_labels = self.data_loader.load_f1_labels()
        self.f1_labels_dict = {(entry['Seriennummer'] , entry['LOGCHARGEDATETIME']): entry for entry in self.f1_labels}

    def load_all_datasets(self):
        """
            load all datasets
            these are loaded and grouped already
        """
        self._load_large_train_data()
        self._load_f1_data()
        self._load_f1_labels()
    