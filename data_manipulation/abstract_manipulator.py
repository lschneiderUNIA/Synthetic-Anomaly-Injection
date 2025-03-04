import pandas as pd
from data_management.data_handler import DataHandler

class AbstractManipulator:
    """
        This is the abstract base class for all data manipulators (anomaly injectors, phase aligners, etc.)
        
        the inject_anomaly function is used on all subclasses to inject an anomaly into the data
    """

    def __init__(self, data_handler : DataHandler):
        self.data_handler = data_handler
        self.align_next_phase = False
        self.align_previous_phase = False

        self.on_selected_phases = None
        self.on_selected_sensors = None

    def __str__(self):
        return f"{self.__class__.__name__}: no subclass"

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            applies the anomaly injection on the DataFrame for given sensor and phase list with parameter dict
            the paramters can be unique for each anomaly type
            to determine the parameters the get_parameter_dict is used 
        """
        raise NotImplementedError("This method is implemented by the subclasses")


    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the anomaly injector
        """
        raise NotImplementedError("This method is implemented by the subclasses")
    

    def requires_alignment_of_next_phase(self):
        """
            return align_next_phase bool
        """
        return self.align_next_phase
    
    def requires_alignment_of_previous_phase(self):
        """
            returns align_previous_phase bool
        """
        return self.align_previous_phase
    
    def get_on_selected_phases(self) -> bool:
        """
            return True if the manipulator works on selected phases or on full data
        """
        if self.on_selected_phases is None:
            raise ValueError("on_selected_phases not set")
        else:
            return self.on_selected_phases


    def get_on_selected_sensors(self) -> bool:
        """
            return True if the manipulator works on selected sensors or on full data        
        """
        if self.on_selected_sensors is None:
            raise ValueError("on_selected_sensors not set")
        else:
            return self.on_selected_sensors