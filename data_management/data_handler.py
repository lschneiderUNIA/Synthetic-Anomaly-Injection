import pandas as pd

from data_management.data_loader import DataLoader

class DataHandler():
    """
        class to handle panda dataframes
        data loading and data manipulation,
        data grouping, all access and write operations

        currently only supports the large train data 
    
    """

    def __init__(self):
        self.data_loader = DataLoader()
        #self.load_all_datasets(self)

        self._load_large_train_data()

        self.list_of_pd_groups = list(self.large_train_data_grouped.groups)
        self.number_of_groups = len(self.list_of_pd_groups)

    def get_number_of_groups(self) -> int:
        """
            return number of groups
        """
        return self.number_of_groups
    
    def get_group(self, group_index : int) -> pd.DataFrame:
        """
            return a group by index
        """
        return self.large_train_data_grouped.get_group(self.list_of_pd_groups[group_index])
    

    
        
    

    def _load_large_train_data(self) -> pd.DataFrame:
        """
            load large train data
        """
        self.large_train_data = self.data_loader.load_large_train_data()
        self.large_train_data_grouped = self.groupby_date_serial_number(self.large_train_data)

        assert self.large_train_data_grouped is not None, "Grouped data is None"
        assert isinstance(self.large_train_data_grouped, pd.core.groupby.generic.DataFrameGroupBy), "Grouped data is not a DataFrameGroupBy object"


    @staticmethod
    def groupby_date_serial_number(data_set : pd.DataFrame) -> pd.DataFrame:
        """
            necessary grouping of data to get individual time series
            groupby does not create a deep copy
        """
        grouped_data = data_set.groupby(['LOGCHARGEDATETIME', 'Seriennummer'])
        return grouped_data

    
   
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
    