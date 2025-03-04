
import pandas as pd
import numpy as np
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging


class PointAnomalyInserter(AbstractManipulator):

    def __init__(self, data_handler : DataHandler):
        super().__init__(data_handler)
        self.on_selected_phases = False
        self.on_selected_sensors = True
        # add new parameters here
        

    def __str__(self):
        return f"PointAnomalyInserter: {round(self.number_of_points_percentage, 5)} | points: {self.number_of_points}"

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            
        """
        phase_length = len(data)
        self.number_of_points = int(phase_length * self.number_of_points_percentage)+1
        anomaly_factors = [
            DataUtilityClass.get_anomaly_factor() for i in range(self.number_of_points)
        ]
        non_null_data = data[pd.notnull(data[sensor])]

        for _, factor in zip(range(self.number_of_points), anomaly_factors):
            random_index = non_null_data.sample().index[0]
            data.at[random_index, sensor] = data.at[random_index, sensor] * factor
        return data

    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the data manipulator
        """
        max_points_in_10k_data_points = 5
        min_points_in_10k_data_points = 1
        max_percentage = max_points_in_10k_data_points/10000
        min_percentage = min_points_in_10k_data_points/10000
        self.number_of_points_percentage = DataUtilityClass.get_random_float(min_percentage, max_percentage)


