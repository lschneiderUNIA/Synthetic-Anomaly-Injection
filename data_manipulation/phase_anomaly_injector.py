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


    def simple_multiply(data, sensor, phase_index, anomaly_factor) -> pd.DataFrame:
        """
            multiply phase data with anomaly factor
        """
        data.loc[data.phase == phase_index, sensor] *= anomaly_factor
        return data
