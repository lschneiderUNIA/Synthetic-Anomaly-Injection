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
        return input_list[start:end]
    

    @staticmethod
    def get_random_elements(list, max_number_of_elements) -> list:
        """
            get a random number of elements from a list

            @param
                list : list -- list
                number : int -- number of elements to get
        """
        set_input = set(list)
        if len(set_input) == 0 or len(set_input) != len(list):
            logging.warning("List has duplicates or is empty")

        if max_number_of_elements > len(list):
            raise ValueError("number of elements to get is larger than the list")
        
        number = random.randint(1, max_number_of_elements)
        return random.sample(set_input, number)
    
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
                         exclude : list = None) -> float:
        """
            get a random float between min and max

            @param
                min : float -- minimum value
                max : float -- maximum value
                precision : int -- precision of the float (ie just round it to that many decimal places)
                exclude : list -- list of values to exclude (we want to exclude 1 for example as anomaly factors)
        """
        if precision:
            increments = 10**-precision
            value_list = np.arange(min, max+increments, increments)
            if exclude:
                value_list = [round(x, precision) for x in value_list]
                value_list = [x for x in value_list if x not in exclude]
            #logging.debug("value_list: {}".format(value_list))

            return random.choice(value_list)
        else:
            while True:
                value = random.uniform(min, max)
                if value not in exclude:
                    return value
