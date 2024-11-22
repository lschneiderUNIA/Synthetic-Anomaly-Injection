import pandas as pd
import numpy as np
import logging
from data_management.data_handler import DataHandler

from data_visualizer import DataVisualizer


class ByGradientAligner():
    """
    This is used to align phases by using inherent steep gradients in the data
    or inserting steep gradients to align phases
    representing fast rise/falls in sensor data

    in the original rational data, this occured quite often
    
    """


    def __init__(self, data_handler : DataHandler) -> None:
        """
            setup data handler
            and parameters for the aligner
            
            minimal_sequence_length is the minimum length of a sequence of consecutive indices in the gradient list
                currently we use 3, but I may experiment with 2 
            desired_gradient is the minimum gradient we search for in the gradient list
                currently 1 (gets changed the -1 if we need a decrease), may experiment with different values as well
        """

        self.data_handler : DataHandler = data_handler


        # 3 is the current minimum sequence length, 
        self.minimal_sequence_length = 3
        self.desired_gradient = 1

    def _find_consecutive_indices(self, input_list : list) -> list:
        """
            finds consecutive values in a list
            we use it to find consecutive indices in the gradient list
            returns the first sequence with length >= self.minimal_sequence_length
        """
        sequence = []
        running_sequence = []
        for iter in range(len(input_list)-1):
            index = input_list[iter]
            if index == input_list[iter+1]-1:
                if len(running_sequence) == 0:
                    running_sequence.append(index)
                running_sequence.append(index+1)
            else:
                sequence = running_sequence
                running_sequence = []
                if len(sequence) >= self.minimal_sequence_length:
                    break

        return sequence


    def align_by_gradient(self, 
                               data : pd.DataFrame, 
                               sensor : str, 
                               phase_index_list : list,
                               align_to_next : bool,
                               align_to_previous : bool
                               ) -> pd.DataFrame:
        """
            main interface for this class
            finds gradient in the phase_index_list and aligns to the previous or following phase depending on align_to_next

            @param
                data : pd.DataFrame -- the data to be changed
                sensor : str -- the sensor to be changed
                phase_index_list : list -- the phases where we search for the gradient
                align_to_next : bool -- align to next or previous phase

        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)

        sensor_data = data[mask]

        desired_gradient = self.desired_gradient

        if align_to_next and align_to_previous:
            logging.error("By-Gradient Aligner: align_to_next and align_to_previous are both True, we handle these seperately")
            return data, False

        if align_to_next:
            logging.info("aligning not implemented")
            return data, False
        else: # align to previous phase

            # get points where phase changes
            phase_change_point = sensor_data.index[0] 
            previous_phase_end_point = phase_change_point - 1

            #values at points
            phase_change_point_value = data.loc[data.index == phase_change_point][sensor].iloc[0]
            previous_phase_value = data.loc[data.index == previous_phase_end_point][sensor].iloc[0]

            # if the phase change point is higher than the previous phase, we need to find a negative gradient since we need to decrease the data
            if previous_phase_value > phase_change_point_value:
                desired_gradient = -desired_gradient
            # else find positive gradient
            
            gradients_of_phases = np.gradient(sensor_data[sensor])
            gradient_index = np.where(gradients_of_phases <= desired_gradient)
            gradient_index = gradient_index[0]

            
        # seconds_index = sensor_data['seconds'].iloc[gradient_index]
        # used this during implementation to test, maybe useful again in the future
        # data_visualizer = DataVisualizer((1,1))
        # data_visualizer.plot_at_grid_position_with_distinct_points(grid_position=(0,0),
        #                                 data= data,
        #                                 x_column='seconds',
        #                                 y_column=sensor,
        #                                 distinct_points=seconds_index,
        #                                 plot_color='red')
        
        # find consecutive indices in the list
        sequence = self._find_consecutive_indices(gradient_index)
        
        # if we cant find a sequence, we cant align
        if len(sequence) < self.minimal_sequence_length:
            print("No sequence found")
            print(sequence)
            return data, False
    
        # note: sequence is inclusive

        # get corresponding indices in the data
        sequence_in_data = sensor_data.iloc[sequence].index
        print("seq: " ,sequence)

        

        # next we need to change the values from the beginning of the phase to the start of the sequence, so it aligns with the previous phase
        #diff at phase change points
        if align_to_next:
            pass # not implemented
        else:
            diff = previous_phase_value - phase_change_point_value
            print("diff: ", diff)
            mask = data.index.isin(range(phase_change_point, sequence_in_data[0] + 1))
            data.loc[mask, sensor] = data.loc[mask][sensor] + diff
            del mask

        start_of_sequence = (sequence_in_data[0], data.iloc[data.index == sequence_in_data[0]][sensor].iloc[0])
        end_of_sequence = (sequence_in_data[-1], data.iloc[data.index == sequence_in_data[-1]][sensor].iloc[0])

        # these are some special cases, where we cant align as easily
        # TODO: implement different strategies for these cases
        if not align_to_next and end_of_sequence[1] < previous_phase_value:
            logging.info("By-Gradient Aligner: end of sequence is lower than previous phase value")
            return data, False


        return self._change_values_by_linear_function(data, sensor, sequence_in_data), True

        


        """
            i used the following code to visualize the gradient and the points where we change the data
            maybe I will use it again in the future
        """

        # data['gradients'] = gradient

        # data_visualizer.plot_at_grid_position_with_distinct_points(grid_position=(0,0),
        #                                 data= data,
        #                                 x_column='seconds',
        #                                 y_column=sensor,
        #                                 distinct_points=[start_of_sequence[0], end_of_sequence[0]],
        #                                 plot_color='red')

        # data_visualizer.show_data()

        return data, True

        

        
    def _change_values_by_linear_function(self, data : pd.DataFrame, sensor : str, sequence_in_data : pd.Index) -> pd.DataFrame:
        """
            changes the values in the data from the start of the sequence to the end of the sequence
            by using a linear function
        """
        # get the start and end of the sequence, including the values at these points
        start_of_sequence = (sequence_in_data[0], data.iloc[data.index == sequence_in_data[0]][sensor].iloc[0])
        end_of_sequence = (sequence_in_data[-1], data.iloc[data.index == sequence_in_data[-1]][sensor].iloc[0])

        # use start_of_sequence and end_of_sequence for a linear function
        # TODO: maybe use different functions or make it not perfectly linear
        slope = (end_of_sequence[1] - start_of_sequence[1])/(end_of_sequence[0] - start_of_sequence[0])
        intercept = start_of_sequence[1] - slope * start_of_sequence[0]
        lin_fun = lambda x : slope * x + intercept

        # get the new values in the sequence
        new_values = [lin_fun(x) for x in range(start_of_sequence[0], end_of_sequence[0] + 1)]
        
        # replace the values
        mask = data.index.isin(range(start_of_sequence[0], end_of_sequence[0] + 1))
        data.loc[mask, sensor] = new_values

        return data, True
