import random
import numpy as np
import logging
class DataUtilityClass():
    """"
        A utility class that has static methods to help with data manipulation
        these functions are used across different classes, thus this central class seems sensible
    
    
    """
    def __init__(self):
        pass

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
                input_list : list -- list 
        """

        start = random.randint(0, len(input_list)-1)
        end = random.randint(start, len(input_list))
        return input_list[start:end]
    

    @staticmethod
    def get_random_elements(list, max_number_of_elements) -> list:
        """
            get a random number of elements from a list

            @param
                list : list -- list
                number : int -- number of elements to get
        """
        if max_number_of_elements > len(list):
            raise ValueError("number of elements to get is larger than the list")
        
        number = random.randint(1, max_number_of_elements)
        return random.sample(list, number)
    
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
            value_list = np.arange(min, max, 10**-precision)
            if exclude:
                value_list = [x for x in value_list if x not in exclude]
            logging.DEBUG("value_list: {}".format(value_list))

            return round(random.choice(value_list), precision)
        else:
            while True:
                value = random.uniform(min, max)
                if value not in exclude:
                    return value
