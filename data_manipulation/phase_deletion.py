

import pandas as pd
import numpy as np
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging


class PhaseDeletion(AbstractManipulator):

    def __init__(self, data_handler : DataHandler):
        super().__init__(data_handler)
        self.on_selected_phases = None
        self.on_selected_sensors = None
        # add new parameters here
        

    def __str__(self):
        pass

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]

        

    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the data manipulator
        """
        self.align_next_phase = True
        self.align_next_phase = True


