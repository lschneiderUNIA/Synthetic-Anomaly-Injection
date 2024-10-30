import pandas as pd
import numpy as np
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass


class PhaseFunctionAnomaly():
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



    def inject_function_on_data(self, 
                               function_parameters : dict,
                               data : pd.DataFrame, sensor : str, 
                               phase_index_list : list, 
                               alignment_factor = None,
                               align_to_next = None) -> pd.DataFrame:
        """
            main interface for this class to apply a function to a set of phases
            also allows for aligning phases

            @param
                function_parameters : dict  -- contains the function type and the parameters for the function
                data : pd.DataFrame -- the data to be changed
                sensor : str -- the sensor to be changed
                phase_index_list : list -- the phases to be changed
                alignment_factor : float -- factor to align the phases, acts as a boolean. If not set we do not align and just change the phases
                align_to_next : bool -- if we align, we have to know if we align to the next or previous phase

        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]
        phase_length = len(phase_data)

        start_index = phase_data.index[0]
        end_index = phase_data.index[-1]

        assert end_index - start_index +1 == phase_length, "Indexing error for length of phases"

        if alignment_factor is not None:
            """
                if alignment factor is set, we have to align the phase_index_list to the previous or next phase
                this means we have to calculate the initial alignment and the phase intersection and then apply the function
                between the start and end index -- which also have to be changed according to the alignment factor
            """
            if align_to_next is None:
                raise ValueError("if alignment factor is set, align_to_next must be set as well")
            if function_parameters['type'] == 'constant':
                raise ValueError("cannot align with constant function") 
            
            phase_length = int(phase_length * alignment_factor)

            # the initial alignent is the ratio between the 2 data points where the phases change
            initial_alignment = self._get_initial_alignment(data, sensor, phase_index_list, align_to_next) 

            if align_to_next:
                """
                    if we align to the next phase, we have to change the start index according to the alignment factor ie "new phase length"
                    the end index stays the same
                """

                function_parameters['start_factor'] = 1
                function_parameters['end_factor'] = initial_alignment
                start_index = end_index - phase_length +1
                if end_index - start_index +1 != phase_length:
                    raise ValueError("indexing calculation error")
            else:
                """
                    analog to the above, just with the start index staying the same and the end index changing
                """
                function_parameters['start_factor'] = initial_alignment
                function_parameters['end_factor'] = 1
                end_index = start_index + phase_length - 1 
                if end_index - start_index + 1  != phase_length:
                    raise ValueError("indexing calculation error")
                
        
        multiplier_list = self._get_multiplier_list(function_parameters, phase_length)     

        return self._apply_function(data, sensor, start_index, end_index, multiplier_list)




    def _get_multiplier_list(self, function_parameters, phase_length):
        """
            get a list of multipliers for the chosen function
            this works off of the function parameters
            for not constant function, we calculate the multipliers in between start and end factor
            with the increase/decrease according to the function type
            @param
                function_parameters : dict -- the function parameters
                phase_length : int -- the length of the phase

        """
        function_name = function_parameters['type']

        if function_name == "constant":
            factor  = function_parameters['factor']
            multiplier_list = [factor for i in range(phase_length)]
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
        

        start_factor  = function_parameters['start_factor']
        end_factor = function_parameters['end_factor']

        # compute initial multiplier list
        multiplier_list = [fun(x) for x in x_points]

        # use the utility class to scale the list of values
        return DataUtilityClass.scale_list_of_values(multiplier_list, start_factor, end_factor)

        

    def _get_initial_alignment(self, data : pd.DataFrame, sensor : str, phase_index_list : list, align_to_next : bool):
        """
            calculate the ratio between the datapoints where the phases change
            this is called the initial alignment

            we have 2 cases depending on the align_to_next parameter, both cases work very similar, just with different indices

            @param
                data : pd.DataFrame -- the data
                sensor : str -- the sensor
                phase_index_list : list -- the phases
                align_to_next : bool -- if we align to the next or previous phase
        """
        if align_to_next:
            # get the last phase
            phase = phase_index_list[-1]
            if phase not in self.data_handler.get_phase_indices_list():
                raise ValueError("phase index not in data")

            # get the last data point of the aligned phases
            # and the next data point which lies in the next phase
            phase_data = data.loc[data['phase'] == phase]
            alignment_index = phase_data.index[-1]
            alignment_index_next_phase = alignment_index + 1
            
            # get the values of the data points
            # the loc returns a series with 1 value, with iloc we get the value itself
            aligment_value = data.loc[data.index == alignment_index][sensor].iloc[0]
            print('alignment value ', aligment_value)
            aligment_value_next = data.loc[data.index == alignment_index_next_phase][sensor].iloc[0]

            ratio = aligment_value_next / aligment_value

        else:
            # meaning align to the previous phase
            phase = phase_index_list[0]
            if phase not in self.data_handler.get_phase_indices_list():
                raise ValueError("phase index not in data")
            
            phase_data = data.loc[data['phase'] == phase]
            alignment_index = phase_data.index[0]       
            alignment_index_previous_phase = alignment_index - 1

            aligment_value = data.loc[data.index == alignment_index][sensor].iloc[0]
            aligment_value_previous = data.loc[data.index == alignment_index_previous_phase][sensor].iloc[0]

            ratio = aligment_value_previous / aligment_value 

        return ratio