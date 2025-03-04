import pandas as pd
import numpy as np
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging


class NoiseInsertion(AbstractManipulator):
    """
        class to insert noise into the data
        TODO: add more noise types, see gits
    """

    def __init__(self, data_handler : DataHandler):
        super().__init__(data_handler)
        self.on_selected_phases = False
        self.on_selected_sensors = True
        # add new parameters here
        self.all_noise_types = ["gaussian"]
        

    def __str__(self):
        return f"NoiseInsertion | {self.noise_type} | {self.max_noise_level_percentage} | {round(self.max_noise_level,3)} | Frequency: {self.frequency}"
        

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            applies noise
            generates noise based on function and frequency
        """
        data_length = len(data)
        data_midrange = (data[sensor].max() + data[sensor].min()) / 2
        self.max_noise_level = data_midrange * self.max_noise_level_percentage
        
        if self.noise_type == "gaussian":
            noise = [
                (i % self.frequency == 0) * self.max_noise_level * random.gauss(0,1)
                for i in range(data_length)
            ]
            data[sensor] = data[sensor] + noise
        else:
            logging.error("Noise type not implemented")
            raise NotImplementedError("Noise type not implemented")

        return data
        

    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the data manipulator

            max_noise_level_percentage : float, maximum noise as the percentage of the midrange
            noise_type : string, type of noise to be added
            frequency : int, how often the noise is added

        """
        self.max_noise_level_percentage = DataUtilityClass.get_random_float(0.001,0.04)
        self.noise_type = DataUtilityClass.get_one_random_element(self.all_noise_types)
        self.frequency = random.randint(1, 6)