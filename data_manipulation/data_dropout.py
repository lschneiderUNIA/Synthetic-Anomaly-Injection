import pandas as pd
import numpy as np
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging


class DataSectionDropout(AbstractManipulator):

    def __init__(self, data_handler : DataHandler):
        super().__init__(data_handler)
        # add new parameters here
        self.on_selected_phases = False
        self.on_selected_sensors = True
        self.sub_type_choices = ["current"] #, "null", "zero"]

        

    def __str__(self):
        return f"DataDropout: {self.dropout_sub_type} |  {round(self.dropout_percentage,2)}"

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            
        """
        all_phases = self.data_handler.get_phase_indices_list()
        mask = self.data_handler.get_mask_for_phases(data, all_phases)
        all_data = data.loc[mask]
        length = len(all_data)

        # get the number of data points to drop
        dropout_length = int(length * self.dropout_percentage)
        dropout_start = random.randint(0, length - dropout_length)

        # get the data points to drop
        dropout_index_mask = all_data.iloc[dropout_start:dropout_start + dropout_length].index

        if self.dropout_sub_type == "current":
            current_value = data.loc[dropout_index_mask[0], sensor]
            data.loc[dropout_index_mask, sensor] = current_value
        elif self.dropout_sub_type == "null":
            data.loc[dropout_index_mask, sensor] = None
        elif self.dropout_sub_type == "zero":
            data.loc[dropout_index_mask, sensor] = 0
        else:
            raise ValueError("Unknown sub type for dropout")
        return data

    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the data manipulator
        """
        self.dropout_percentage = DataUtilityClass.get_random_float(0, 1)
        self.dropout_sub_type = random.choice(self.sub_type_choices)

            

