import pandas as pd
import numpy as np
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging

class PhaseFunctionAnomaly(AbstractManipulator):
    """
    Base class for using different functions to change a set of phases
    also allows for aligning phases, since its bascically the same thing
    in practice, we dont go use phases, but start and end points since the alignment doesn't always change the full phase

    works as a standard interface     
    """

    def __init__(self, data_handler : DataHandler) -> None:
        """
            not yet sure if we need an init or all of these functions can be static
        """
        self.data_handler : DataHandler = data_handler

        self.all_function_types = ['constant', 'linear', 'exponential', 'arctan']

        self.function_types_for_alingment = ['exponential', 'arctan']

    def _apply_function(self, data, sensor, start_index, end_index, multiplier_list):
        """
            apply function to a set of data points
            @param
                data : pd.DataFrame
                sensor : str
                start_index : int
                end_index : int
                multiplier_list : list of floats with same length as start to end_index
                    is based on the chosen function beforehand
        """
        data.loc[start_index:end_index, sensor] *= multiplier_list
        return data
    
    def __str__(self):
        return f"PhaseFunctionAnomaly, {self.function_type}, align factor {self.alignment_factor}, start/end {self.start_factor}/{self.end_factor}"
    
    def set_manipulation_parameters(self, align_next_phase : bool = False, align_previous_phase : bool = False):
        """
            sets the internal parameters for the anomaly injector

            we have to handle the bool inputs of course,
            if they aren't set we simply inject an anomaly, in this case we have to set internally if we need alignment

            we need:
                function_type
                    factor for constant
                start_factor
                end_factor
                alignment_factor (optional)                
                align bools if current call is for simple anomaly injection
        """

        if align_next_phase and align_previous_phase:
            raise ValueError("cannot align to next and previous phase at the same time")
        
        
        if align_next_phase or align_previous_phase:
            # if we align, we need a alignment factor
            # and need a compatible function
            self.alignment_factor = DataUtilityClass.get_random_float(min = 0.4, 
                                                                      max = 0.7, 
                                                                      precision = 1)
            
            self.function_type = DataUtilityClass.get_one_random_element(self.function_types_for_alingment)
            self.align_next_phase = align_next_phase
            self.align_previous_phase = align_previous_phase
        else:
            # if we dont align, we can use any function
            # and need to set the alignment factor to None
            # also need to pick start and end factor
            # and determine alignment bools

            self.function_type = DataUtilityClass.get_one_random_element(self.all_function_types)
            self.alignment_factor = None

            if self.function_type == 'constant':
                # this is a special case, we only need one factor
                self.factor = self._get_factor()
                self.align_next_phase = True
                self.align_previous_phase = True
                return

            # 80% of the time we set one of start and end to 1.0
            # 20% of the time we set both to a random factor     
            # if we set both to a random factor, we have to align to both neighbouring phases     
            if random.random() < 0.8:
                if random.random() < 0.5:
                    self.start_factor = 1.0
                    self.end_factor = self._get_factor()
                    self.align_next_phase = True
                    self.align_previous_phase = False
                else:
                    self.end_factor = 1.0
                    self.start_factor = self._get_factor()
                    self.align_next_phase = False
                    self.align_previous_phase = True
            else:
                self.start_factor = self._get_factor()
                self.end_factor = self._get_factor()
                self.align_next_phase = True
                self.align_previous_phase = True

            self.anomaly_injection_start = self.start_factor
            self.anomaly_injection_end = self.end_factor
            
        
    def _get_factor(self):
        """
            returns a start or end factor for the function
            avoids having to call the same get_random_float multiple times
            
            we can also handle the ranges in here
        """
        min_factor = 0.5
        max_factor = 1.5

        return DataUtilityClass.get_random_float(min = min_factor, 
                                                 max = max_factor, 
                                                 precision = 1, 
                                                 exclude = [0.9, 1.0, 1,1])

    
    def apply_manipulation(self, 
                       data : pd.DataFrame, 
                       sensor : str, 
                       phase_index_list : list) -> pd.DataFrame:
        """
            main interface for this class to apply a function to a set of phases
            also allows for aligning phases
            if self.align_to_next is True, we get the next phase in phase_index_list from the original anomaly injection
                thus, the phase change is between phase_index_list and the phase before
                

            @param
                data : pd.DataFrame -- the data to be changed
                sensor : str -- the sensor to be changed
                phase_index_list : list -- the phases to be changed
                alignment_factor : float -- factor to align the phases, acts as a boolean. If not set we do not align and just change the phases

        """
        if self.align_next_phase and self.align_previous_phase and self.alignment_factor is not None:
            raise ValueError("cannot align to next and previous phase at the same time")

        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]
        phase_length = len(phase_data)

        start_index = phase_data.index[0]
        end_index = phase_data.index[-1]

        assert end_index - start_index + 1 == phase_length, "Indexing error for length of phases"

        if self.alignment_factor is not None:
            """
                if alignment factor is set, we have to align the phase_index_list to the previous or next phase
                this means we have to calculate the initial alignment and the phase intersection and then apply the function
                between the start and end index -- which also have to be changed according to the alignment factor
            """
            if self.function_type == 'constant':
                raise ValueError("cannot align with constant function") 
            
            phase_length = int(phase_length * self.alignment_factor)

            #the initial alignent is the ratio between the 2 data points where the phases change
            initial_alignment = DataUtilityClass.get_phase_change_ratio(data, 
                                                                        sensor,
                                                                        phase_index_list,
                                                                        self.align_next_phase,
                                                                        self.align_previous_phase) 

            if self.align_next_phase:
                """
                    if we align to the next phase, we have to change the start index according to the alignment factor ie "new phase length"
                    the end index stays the same
                """
                self.start_factor = initial_alignment
                self.end_factor = 1
                end_index = start_index + phase_length - 1 
                if end_index - start_index +1 != phase_length:
                    raise ValueError("indexing calculation error")  
            elif self.align_previous_phase:
                """
                    analog to the above, just with the start index staying the same and the end index changing
                """
                self.start_factor = 1
                self.end_factor = initial_alignment
                start_index = end_index - phase_length +1
                
                if end_index - start_index + 1  != phase_length:
                    raise ValueError("indexing calculation error")
            else:
                raise ValueError("alignment factor is set, but align_next_phase is not set")
        
        multiplier_list = self._get_multiplier_list(phase_length)     

        return self._apply_function(data, sensor, start_index, end_index, multiplier_list)




    def _get_multiplier_list(self, phase_length):
        """
            get a list of multipliers for the chosen function
            this works off of the function parameters
            for not constant function, we calculate the multipliers in between start and end factor
            with the increase/decrease according to the function type
            @param
                function_parameters : dict -- the function parameters
                phase_length : int -- the length of the phase

        """
        function_name = self.function_type

        if function_name == "constant":
            multiplier_list = [self.factor for i in range(phase_length)]
            return multiplier_list
        
        elif function_name == "linear":
            x_points = np.linspace(0, 1, phase_length)
            fun = lambda x : x
            # start_factor  = function_parameters['start_factor']
            # end_factor = function_parameters['end_factor']
            #multiplier_list = [start_factor + (end_factor-start_factor)*i/phase_length for i in range(phase_length)]
                    
        elif function_name == "exponential":
            # this is the input range for the exponential function
            # I purley picked them on feel, I suspect they irrelavant, as it is just a scaling factor and the relative growth is the same
            x_points = np.linspace(-3, 3, phase_length)
            fun = lambda x : np.exp(x) 
        
        elif function_name == "arctan":
            x_points = np.linspace(-4, 4, phase_length)
            fun = lambda x : np.arctan(x)
        else:
            raise ValueError("function name not recognized")
        
        # compute initial multiplier list
        multiplier_list = [fun(x) for x in x_points]

        # use the utility class to scale the list of values
        return DataUtilityClass.scale_list_of_values(multiplier_list, self.start_factor, self.end_factor)
    

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