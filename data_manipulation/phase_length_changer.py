import pandas as pd
import numpy as np
import scipy.interpolate
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging
import scipy


class PhaseLengthChanger(AbstractManipulator):

    """
        this class stretches or shortens the length of a set of phases
        this means we have to change all sensor values and the seconds column
        we only change the relevant columns set in the options file

        we interpolate the data to get the new values
    """

    def __init__(self, data_handler : DataHandler):
        super().__init__(data_handler)
        self.on_selected_phases = True
        self.on_selected_sensors = False
        # add new parameters here
        

    def __str__(self):
        return f"PhaseLengthChanger: {self.anomaly_factor} | true factor: {round(self.true_factor,5)}"

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            we create a new pandas frame with the stretched or shortened phases
            this will look like this
                data before phase_index_list
                individual phases in input list stretched or shortened by building a dict
                remaining data after phase_index_list, including shift of seconds

        """
        data = data[self.data_handler.get_relevant_columns()].copy() # we only consider relevant columns

        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]
        phase_length = len(phase_data)

        # construct new phase data
        stretched_phase_dict = {}

        # we have to stretch each phase individually
        # multiplying length by anomaly factor, may not make it possible to stretch all phases according to original proportions
        individual_phase_lengths = [
            len(phase_data[phase_data['phase'] == i]) for i in phase_index_list
        ]
        new_individual_phase_lengths = [
            int(length * self.anomaly_factor) for length in individual_phase_lengths
        ]

        # sum of new phases and the "true" factor
        new_phases_length = sum(new_individual_phase_lengths)
        self.true_factor = new_phases_length / phase_length
        logging.debug(f"True/set factor: {self.true_factor} -- {self.anomaly_factor}")
        logging.debug(f"New phase length: {new_phases_length} -- Old phase length: {phase_length}")
        diff_length =  new_phases_length - phase_length

        # phase columns also has to change
        new_phase_column = [
            length * [i] for i, length in zip(phase_index_list, new_individual_phase_lengths)
        ]
        new_phase_column = [item for sublist in new_phase_column for item in sublist] # flatten list
        stretched_phase_dict['phase'] = new_phase_column

        # seconds just needs a new range according to new_phases_length
        pre_phases_seconds = phase_data['seconds'].iloc[0]
        new_seconds_column = range(pre_phases_seconds, pre_phases_seconds + new_phases_length)
        stretched_phase_dict['seconds'] = new_seconds_column

        # Interpolation: 
        # we act as if the data is in range new_phases_length, but not integer values (i * true_factor can be float)
        old_range = [
            i * self.true_factor for i in range(phase_length) 
        ]
        new_range = np.linspace(0, new_phases_length, new_phases_length)

        # interpolate for all sensors
        for sensor in self.data_handler.get_data_sensor_list():
            new_data_points = np.interp(new_range, old_range, phase_data[sensor])
            stretched_phase_dict[sensor] = new_data_points

        # shift seconds of remaining phases
        remaining_data = data.loc[data.index > phase_data.index[-1]]  
        remaining_data.loc[:,'seconds'] += diff_length
        
        new_data_frame = pd.concat([data.loc[data.index < phase_data.index[0]],
                                    pd.DataFrame(stretched_phase_dict),
                                    remaining_data],
                                    ignore_index=True)
            
        
        return new_data_frame

       



    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the data manipulator
        """
        self.anomaly_factor = DataUtilityClass.get_anomaly_factor()

