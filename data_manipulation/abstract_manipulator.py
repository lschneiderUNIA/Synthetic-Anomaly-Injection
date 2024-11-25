import pandas as pd

class AbstractManipulator:
    """
        This is the abstract base class for all data manipulators (anomaly injectors, phase aligners, etc.)
        
        the inject_anomaly function is used on all subclasses to inject an anomaly into the data
    """

    def __init__(self, data_handler):
        self.data_handler = data_handler

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            applies the anomaly injection on the DataFrame for given sensor and phase list with parameter dict
            the paramters can be unique for each anomaly type
            to determine the parameters the get_parameter_dict is used 
        """
        pass


    def set_manipulation_parameters(align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the anomaly injector
        """
        pass
    
    def requires_alignment_of_next_phase(self) -> bool:
        """
            return align_to_next bool
        """
        pass


    def requires_alignment_of_previous_phase(self) -> bool:
        """
            returns align_to_previous bool
        """
        pass