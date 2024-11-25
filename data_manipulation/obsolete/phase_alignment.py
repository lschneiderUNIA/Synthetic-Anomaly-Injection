import pandas as pd
import numpy as np

from data_management.data_handler import DataHandler


class PhaseAligner():
    """
        PhaseAlignment class is used to align surrounding phases after an anomaly injection has been done 
        that changes the start/end values of a phase
    
    """


    def __init__(self, data_handler : DataHandler) -> None:
        
        self.data_handler = data_handler
        self.phase_indices_list : np.ndarray = self.data_handler.get_phase_indices_list()


    


    def _get_phase_shift_ratios(self, alignment_type,
                                 is_previous : bool,
                                 phase_shift_length : int,
                                 phase_shift_ratio) -> np.ndarray:
        """
            get phase shift ratios to align phases

            alignment_type : str
                type of alignment
            is_previous : bool  
                if the previous phase is to be aligned, if false the next phase is aligned
        """
        if alignment_type == 'linear':
            if is_previous:
                return np.linspace(1.0, phase_shift_ratio, phase_shift_length)
            else:
                return np.linspace(phase_shift_ratio, 1.0, phase_shift_length)
        elif alignment_type == 'log':
            if is_previous:
                return np.logspace(np.log10(1.0), np.log10(phase_shift_ratio), phase_shift_length)
            else:
                return np.logspace(np.log10(phase_shift_ratio),  np.log10(1.0), phase_shift_length)
        elif alignment_type == 'arctan':
            raise ValueError("Needs to be fixed")
            if is_previous:
                return PhaseAligner.arctan_space(1.0, phase_shift_ratio, phase_shift_length)
            else:
                return PhaseAligner.arctan_space(phase_shift_ratio, 1.0, phase_shift_length)
        else:
            raise ValueError("Alignment type not supported")

    

    def phase_alignment(self,
                            alignment_type : str,
                               data : pd.DataFrame, 
                               sensor : str, 
                               changed_phase_index : int, 
                               alingment_phase_length_percentage : float) -> pd.DataFrame:
        """
            align phases linearly

            data : pd.DataFrame
                data to be aligned
            sensor : str
                sensor to be aligned
            changed_phase_index : int
                phase index of the changed phase
            alingment_phase_length_percentage : float
                percentage of the previous phase to be aligned
        """

        # TODO: phase checking is to complicated rn, simplify the if statements

        
        if np.where(self.phase_indices_list == changed_phase_index)[0].size == 0:        
            raise ValueError("Phase index not found in phase_indices_list")
        
        if np.where(self.phase_indices_list == changed_phase_index)[0] != 0:
            """
                this means the changed phase is not the first phase
                and we need to change the previous phase
            """
            previous_phase_index = changed_phase_index - 1
            previous_phase_length = len(data[data.phase == previous_phase_index])
            phase_alignment_length = int(previous_phase_length * alingment_phase_length_percentage)
            previous_phase_data = data.loc[data.phase == previous_phase_index, sensor].iloc[-phase_alignment_length:]

            previous_phase_data_end_value = previous_phase_data.iloc[-1]
            changed_phase_data_start_value = data.loc[data.phase == changed_phase_index, sensor].iloc[0]
            transition_ratio = changed_phase_data_start_value / previous_phase_data_end_value

            data_shift = self._get_phase_shift_ratios(alignment_type, True, phase_alignment_length, transition_ratio)

            changed_values = previous_phase_data * data_shift

            previous_phase_row = data.loc[data.phase == previous_phase_index]

            # Update the sensor column for the first phase_shift_length rows
            data.loc[previous_phase_row.index[-phase_alignment_length:], sensor] = changed_values        

        # check if last phase
        if np.where(self.phase_indices_list == changed_phase_index)[0] != len(self.phase_indices_list) -1:  
            """
                basically the same as for the previous phase
                TODO: maybe there is a way to combine these two
            """
            next_phase_index = changed_phase_index + 1
            next_phase_length = len(data[data.phase == next_phase_index])
            phase_alignment_length = int(next_phase_length * alingment_phase_length_percentage)
            next_phase_data = data.loc[data.phase == next_phase_index, sensor].iloc[:phase_alignment_length]

            next_phase_data_start_value = next_phase_data.iloc[0]
            changed_phase_data_end_value = data.loc[data.phase == changed_phase_index, sensor].iloc[-1]
            transition_ratio = changed_phase_data_end_value / next_phase_data_start_value

            data_shift = self._get_phase_shift_ratios(alignment_type, False, phase_alignment_length, transition_ratio)

            changed_values = next_phase_data * data_shift

            next_phase_row = data.loc[data.phase == next_phase_index]

            # Update the sensor column for the first phase_shift_length rows
            data.loc[next_phase_row.index[:phase_alignment_length], sensor] = changed_values


        return data
    
    @staticmethod
    def arctan_space(start, stop, num):
        """
        Generates `num` points between `start` and `stop` using an inverse tangent distribution.
        
        start: lower bound of the range
        stop: upper bound of the range
        num: number of points to generate (default is 50)
        
        Returns:
        - np.ndarray: Array of `num` values distributed between start and stop with an arctangent curve.
        """

        linear_range = np.arange(num)

        A = (stop - 1) / np.arctan(start)  # Scales to match the range
        B = 2 / num                   # Controls steepness
        C = num / 2                   # Inflection at center
        D = start                     # Shifts to pass through (0,1)

        arctan_fun = lambda x : A * np.arctan(B * (x - C)) + D

        result = arctan_fun(linear_range)
        return result