import pandas as pd
from data_management.data_handler import DataHandler


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
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]
        phase_length = len(phase_data)

        start_index = phase_data.index[0]
        end_index = phase_data.index[-1]

        assert end_index - start_index +1 == phase_length, "Indexing error for length of phases"

        if alignment_factor is not None:
            if align_to_next is None:
                raise ValueError("if alignment factor is set, align_to_next must be set as well")
            if function_parameters['type'] == 'constant':
                raise ValueError("cannot align with constant function") 
            
            phase_length = int(phase_length * alignment_factor)

            initial_alignment = self._get_initial_alignment(data, sensor, phase_index_list, align_to_next) 
            if align_to_next:
                function_parameters['start_factor'] = 1
                function_parameters['end_factor'] = initial_alignment
                start_index = end_index - phase_length +1
                if end_index - start_index +1 != phase_length:
                    print("start index: ", start_index)
                    print("end index: ", end_index)
                    print("e -s: ", end_index - start_index)
                    print("phase length: ", phase_length)
                    raise ValueError("indexing calculation error")
            else:
                function_parameters['start_factor'] = initial_alignment
                function_parameters['end_factor'] = 1
                end_index = start_index + phase_length - 1 
                if end_index - start_index + 1  != phase_length:
                    print("start index: ", start_index)
                    print("end index: ", end_index)
                    print("e -s: ", end_index - start_index)
                    print("phase length: ", phase_length)
                    raise ValueError("indexing calculation error")

        multiplier_list = self._get_multiplier_list(function_parameters, phase_length)     

        return self._apply_function(data, sensor, start_index, end_index, multiplier_list)




    def _get_multiplier_list(self, function_parameters, phase_length):
        """
            get a list of multipliers for the chosen function

            this works off of start and end factors, between which the function will be applied
        """
        function_name = function_parameters['type']
        if function_name == "constant":
            factor  = function_parameters['factor']
            multiplier_list = [factor for i in range(phase_length)]
            return multiplier_list
        elif function_name == "linear":
            start_factor  = function_parameters['start_factor']
            end_factor = function_parameters['end_factor']
            multiplier_list = [start_factor + (end_factor-start_factor)*i/phase_length for i in range(phase_length)]
            return multiplier_list
        else:
            raise ValueError("function name not recognized")
        

    def _get_initial_alignment(self, data : pd.DataFrame, sensor : str, phase_index_list : list, align_to_next : bool):
        """
        
        """
        if align_to_next:
            phase = phase_index_list[-1]
            if phase not in self.data_handler.get_phase_indices_list():
                raise ValueError("phase index not in data")

            phase_data = data.loc[data['phase'] == phase]
            alignment_index = phase_data.index[-1]
            alignment_index_next_phase = alignment_index + 1
            
            # the loc returns a series with 1 value, with iloc we get the value itself
            aligment_value = data.loc[data.index == alignment_index][sensor].iloc[0]
            print('alignment value ', aligment_value)
            aligment_value_next = data.loc[data.index == alignment_index_next_phase][sensor].iloc[0]

            ratio = aligment_value_next / aligment_value

        else:
            # meaning align to the previous phase
            # works very similar to the above, just with different indices
            phase = phase_index_list[0]
            if phase not in self.data_handler.get_phase_indices_list():
                raise ValueError("phase index not in data")
            
            phase_data = data.loc[data['phase'] == phase]
            alignment_index = phase_data.index[0]       
            alignment_index_previous_phase = alignment_index - 1

            aligment_value = data.loc[data.index == alignment_index][sensor].iloc[0]
            aligment_value_previous = data.loc[data.index == alignment_index_previous_phase][sensor].iloc[0]

            ratio = aligment_value / aligment_value_previous

        print('ratio ', ratio)
        print('ratio type ', type(ratio))   
        return ratio