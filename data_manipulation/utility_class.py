import random
import numpy as np
import pandas as pd
import logging
class DataUtilityClass():
    """"
        A utility class that has static methods to help with data manipulation
        these functions are used across different classes, thus this central class seems sensible
    
    
    """
    def __init__(self):
        pass

    @staticmethod
    def get_phase_change_ratio(data : pd.DataFrame, 
                               sensor : str, 
                               phase_index_list : list,
                               align_next_phase : bool,
                               align_previous_phase) -> float:
        """
            calculate the ratio between the datapoints where the phases change
            this is called the initial alignment

            we have 2 cases depending on the align_next_phase parameter, both cases work very similar, just with different indices
            we have assured before this function call, that align_next_phase and align_previous_phase are not both True
            also phase_index_list is in the data

            @param
                data : pd.DataFrame -- the data
                sensor : str -- the sensor
                phase_index_list : list -- the phases
        """
        if align_next_phase and align_previous_phase:
            raise ValueError("Phase Change Ratio: Both align_next_phase and align_previous_phase are True")
        if not align_next_phase and not align_previous_phase:
            raise ValueError("Phase Change Ratio: Both align_next_phase and align_previous_phase are False")
        
        if align_next_phase:
            # get first phase in the list
            phase = phase_index_list[0]

            # get the first data point of the phases to be aligned
            # and the previous data in the phase before
            phase_data = data.loc[data['phase'] == phase]
            phase_data_start_index = phase_data.index[0]
            previous_phase_end_index = phase_data_start_index -1 
            
            # get the values of the data points
            # the loc returns a series with 1 value, with iloc we get the value itself
            phase_data_start_value = data.loc[data.index == phase_data_start_index][sensor].iloc[0]
            previous_phase_end_value = data.loc[data.index == previous_phase_end_index][sensor].iloc[0]

            ratio = previous_phase_end_value / phase_data_start_value

        else:
            # meaning aligning the previous phase of the original anomaly injection
            phase = phase_index_list[-1]
            
            phase_data = data.loc[data['phase'] == phase]
            phase_data_end_index = phase_data.index[-1]       
            next_phase_start_index = phase_data_end_index +1

            phase_data_end_value = data.loc[data.index == phase_data_end_index][sensor].iloc[0]
            next_phase_start_value = data.loc[data.index == next_phase_start_index][sensor].iloc[0]

            ratio = next_phase_start_value / phase_data_end_value 

        return ratio
    
    @staticmethod
    def get_anomaly_factor():
        """
            returns a random float for anomaly factors
            we call this from multiple classes
            avoids having to call the same get_random_float multiple times 
            and we can adjust the ranges here if needed
            kind of assumes that we need the same factors for all anomalies 

        """
        min_factor = 0.5
        max_factor = 1.6

        return DataUtilityClass.get_random_float(min = min_factor, 
                                                 max = max_factor, 
                                                 precision = 2, 
                                                 exclude_min_max = [0.8, 1.2]) # exclude values between 0.8 and 1.2

    @staticmethod
    def scale_list_of_values(value_list, new_min, new_max):
        """
            inputs a list of values and scales them to a new range

            @param
                value_list : list -- list of values
                new_min : float -- new minimum value
                new_max : float -- new maximum value
        
        """
        max_value = max(value_list)
        min_value = min(value_list)

        old_min = min_value
        old_max = max_value

        # formula I found online
        value_list = [new_min + ((x - old_min) / (old_max - old_min)) * (new_max - new_min)
                            for x in value_list]
        # formula: new_min + ((array - old_min) / (old_max - old_min)) * (new_max - new_min)

        return value_list
    

    @staticmethod
    def get_random_slice(input_list) -> list:
        """
            get a random slice of the phase index list
            this is used to get a random set of phases

            @param
                input_list : list
        """

        start = random.randint(0, len(input_list)-1)
        end = random.randint(start+1, len(input_list))
        return input_list[start:end].tolist()
    

    @staticmethod
    def get_random_elements(input_list, max_number_of_elements) -> list:
        """
            get a random number of elements from a list
            list entries must be hashable

            @param
                list : list -- list
                number : int -- number of elements to get
        """
        input_list = list(input_list)
        assert isinstance(input_list, list), f"input is of type {type(input_list)}"
        # set_input = set(list)
        # if len(set_input) == 0 or len(set_input) != len(list):
        #     logging.warning("List has duplicates or is empty")

        if max_number_of_elements > len(input_list):
            raise ValueError("number of elements to get is larger than the list")
        
        number = random.randint(1, max_number_of_elements)
        return random.sample(input_list, number)
    
    @staticmethod
    def get_one_random_element(list):
        """
                get one random element from a list

            @param
                list : list -- list of elements
        """
        return random.choice(list)
    
    @staticmethod
    def get_random_float(min,
                         max, 
                         precision : int = None,
                         exclude_min_max : tuple[int] = None) -> float:
        """
            get a random float between min and max
            exlude 

            @param
                min : float -- minimum value
                max : float -- maximum value
                precision : int -- precision of the float (ie just round it to that many decimal places)
                exclude : int tuple with min and max of excluded values
                    I used a list before, which is dumb when we have higher precision inputs
        """
        sampling_tries = 0
        if exclude_min_max and precision:
            while True:
                if sampling_tries > 10:
                    raise ValueError("Reconfigure get_random_float parameters")
                value = random.uniform(min, max)
                if value > exclude_min_max[0] and value < exclude_min_max[1]:
                    sampling_tries += 1
                else:
                    return round(value, precision)

        else:
            return random.uniform(min, max)


        # tried to do it without precision, but in combi with exclude is stupid

    @staticmethod
    def get_random_bool():
        """
            get a random boolean value
        """
        return random.choice([True, False]) 