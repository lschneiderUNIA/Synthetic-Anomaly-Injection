import pandas as pd
from data_management.data_handler import DataHandler


class PhaseAnomalyInjector():
    """
    Base class for anomaly injection for individual or a set of phases
    I want a standard interface for all anomaly injectors
       
    
    """

    def __init__(self, data_handler : DataHandler) -> None:
        """
            not yet sure if we need an init or all of these functions can be static
        """
        self.data_handler : DataHandler = data_handler


    def linear_function(self, data, sensor, phase_index_list, anomaly_factor) -> pd.DataFrame:
        """
            multiply phase data with anomaly factor
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        data.loc[mask, sensor] *= anomaly_factor
        return data


    def constant_function(self,data, sensor, phase_index_list, anomaly_factor) -> pd.DataFrame:
        """
            reduce phase data by anomaly factor
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        data.loc[mask, sensor] += anomaly_factor
        return data
    
    
    def polynomial_function(self, data, sensor, phase_index_list, anomaly_factor) -> pd.DataFrame:
        """
            apply a quadratic function to phase data
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        data.loc[mask, sensor] = data.loc[mask, sensor]**anomaly_factor
        return data