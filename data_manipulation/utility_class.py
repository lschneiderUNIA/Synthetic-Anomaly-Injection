

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