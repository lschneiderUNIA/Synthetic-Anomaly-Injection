import pandas as pd
import numpy as np
import logging
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass

from data_visualizer import DataVisualizer


class ByGradientAligner():
    """
    This is used to align phases by using inherent steep gradients in the data
    or inserting steep gradients to align phases
    representing fast rise/falls in sensor data

    in the original rational data, this occured quite often
    
    """


    def __init__(self, data_handler : DataHandler,
                 desired_gradient : float = 0.75,
                 minimal_sequence_length : int = 3,
                 max_tries_to_find_sequence : int = 3) -> None:
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
        self.minimal_sequence_length = minimal_sequence_length
        self.desired_gradient = desired_gradient
        self.max_tries_to_find_sequence = max_tries_to_find_sequence

        self.selected_sequence_length = None
        self.selected_gradient = None

    def __str__(self) -> str:
        return f"ByGradientAligner: length {self.selected_sequence_length}, gradient {self.selected_gradient}"

    def _find_consecutive_indices(self,
                                  input_list : list,
                                  allowed_sequence_break = 1) -> list:
        """
            finds consecutive indices values in a list
            we use it to find consecutive indices in the gradient list
            returns the first sequence with length >= self.minimal_sequence_length
            also sorts by length, since longer should be better
        """

        suitable_sequences = []
        running_sequence = []
        for iter in range(len(input_list)-1):
            index = input_list[iter]
            allowed_values = [
                input_list[iter+1] + i for i in range(-1, allowed_sequence_break + 1)
            ]
            if index in allowed_values:
                # value is consecutive or with in the allowed_sequence_break
                if len(running_sequence) == 0:
                    # if we have a new sequence, we need to add 2 indices
                    running_sequence.append(index)
                running_sequence.append(index+1)
            else: # value not consecutive
                # add running sequence to sequence if > minimal_sequence_length
                # return if we have >= self.max_tries_to_find_sequence    
                if len(running_sequence) >= self.minimal_sequence_length:        
                    suitable_sequences.append(running_sequence)
                running_sequence = []
                if len(suitable_sequences) >= self.max_tries_to_find_sequence:
                    break

        # sort sequences by length
        suitable_sequences = sorted(suitable_sequences, key=len, reverse=True)
        return suitable_sequences
    

    def find_gradient_sequence(self, data : pd.DataFrame, sensor : str) -> list:
        """
            finds a sequence of high gradients or low gradients in the data
        """
        # get gradients and subsequent sequences of high/low gradients
        gradients_of_phases = np.gradient(data[sensor])

        positive_gradients = np.where(gradients_of_phases >= self.desired_gradient)
        positive_gradients = positive_gradients[0] # returns a tuple I think with values in the second element I think, we only need indices

        negative_gradients = np.where(gradients_of_phases <= -self.desired_gradient)
        negative_gradients = negative_gradients[0]

        # these will already only return <= max_tries_to_find_sequence, so we can actually check all of them
        # and are sorted by length
        positive_gradients_sequences = self._find_consecutive_indices(positive_gradients)
        negative_gradients_sequences = self._find_consecutive_indices(negative_gradients)

        return positive_gradients_sequences, negative_gradients_sequences

    def align_by_gradient(self, 
                               data : pd.DataFrame, 
                               sensor : str, 
                               phase_index_list : list,
                               align_next_phase : bool,
                               align_previous_phase : bool
                               ) -> pd.DataFrame:
        """
            main interface for this class
            finds gradient in the phase_index_list and aligns to the previous or following phase depending on align_next_phase

            @param
                data : pd.DataFrame -- the data to be changed
                sensor : str -- the sensor to be changed
                phase_index_list : list -- the phases where we search for the gradient
                align_next_phase : bool -- align next phase of original injection
                align_previous_phase : bool -- align previous phase of original injection

        """

        if align_next_phase and align_previous_phase:
            logging.error("By-Gradient Aligner: align_next_phase and align_previous_phase are both True, we handle these seperately")
            return data, False
        

        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        original_phase_data = data[mask]
        working_phase_data = original_phase_data.copy()

        """
            to simplify the 2 cases, we approach them as followed:
                - align_next_phase is our main case and mirror/flip the data if align_previous_phase is True
                - we use a "anomaly_point_value" to which we need to align representing the first/last point 
                    of the anomaly injected phase
                - we use a "to_be_aligned_value" representing the first/last point of the phase we want to align to
                - we still need the original indices to later change the actual data         
                - we use a "working_phase_data" and later assign the new values to the original data
                - but as such we can solve the problem as if we only have one case
                    and mirror the data afterwards again

        """
        if align_previous_phase:
            to_be_aligned_index = working_phase_data.index[-1]
            anomaly_point_index = to_be_aligned_index + 1 

            #values at points
            to_be_aligned_value = data.loc[data.index == to_be_aligned_index][sensor].iloc[0]
            anomaly_point_value = data.loc[data.index == anomaly_point_index][sensor].iloc[0]

            # Sort by 'seconds' in descending order (reverse)
            working_phase_data = original_phase_data.sort_values(by='seconds', ascending=False)

            # Reset the index based on the new order
            working_phase_data = working_phase_data.reset_index(drop=True)
            working_phase_data['original_index'] = original_phase_data.index

        else: # align next phase

            # get points where phase changes
            to_be_aligned_index = working_phase_data.index[0] 
            anomaly_point_index = to_be_aligned_index - 1

            #values at points
            to_be_aligned_value = data.loc[data.index == to_be_aligned_index][sensor].iloc[0]
            anomaly_point_value = data.loc[data.index == anomaly_point_index][sensor].iloc[0]

            # here we don't change the indices, but will still use the original_index column later
            # maybe we dont???
            working_phase_data['original_index'] = original_phase_data.index

        # as in the start index is new since we may have changed the indexing if align previous phase
        new_phase_start_index = working_phase_data.index[0]

        # to reduce the cases even more, we horizonally mirror the data (ie * -1), 
        # such that the anomaly_point_value is always lower than the to_be_aligned_value
        flip_values = False
        if anomaly_point_value > to_be_aligned_value:
            working_phase_data[sensor] = working_phase_data[sensor] * -1
            to_be_aligned_value = to_be_aligned_value * -1
            anomaly_point_value = anomaly_point_value * -1
            flip_values = True

        
        positive_gradients_sequences, negative_gradients_sequences = self.find_gradient_sequence(working_phase_data, sensor)

        # if we cant find a sequence, we cant align
        if len(positive_gradients_sequences) == 0 and len(negative_gradients_sequences) == 0:
            logging.debug("By-Gradient Aligner: Could not find a sequence")
            return data, False
        
        logging.debug("By-Gradient Aligner: Found sequences: positive: {}, negative: {}".format(len(positive_gradients_sequences), len(negative_gradients_sequences)))

        random_sequence_permutation = self._gradient_sequences_sampling(positive_gradients_sequences, negative_gradients_sequences)

        used_sequence = False
        for (sequence, gradient_type) in random_sequence_permutation:

            sequence_in_data = working_phase_data.iloc[sequence].index

            diff = to_be_aligned_value - anomaly_point_value


            if gradient_type == 'positive':      
                # in this case, we simply need to downshift the phase start until the sequence start
                # as we shift against the direction of gradient, we have no issues here
                # and then rescale the sequence values
                # logging.debug("diff: {}, sequence start: {}, phase start: {}".format(diff, 
                #                                                                      working_phase_data.loc[working_phase_data.index == sequence_in_data[0]]['seconds'], 
                #                                                                      working_phase_data.loc[working_phase_data.index == new_phase_start_index]['seconds']))

                phase_start_mask = working_phase_data.index.isin(range(new_phase_start_index, sequence_in_data[0]))   
                working_phase_data.loc[phase_start_mask, sensor] = working_phase_data.loc[phase_start_mask][sensor] - diff
                del phase_start_mask # idk why I do this

                sequence_mask = working_phase_data.index.isin(sequence_in_data)
                if len(sequence_in_data) != len(working_phase_data.loc[sequence_mask]):
                    raise ValueError("By-Gradient Aligner: Sequence length does not match")

                # get the values at the start and end of the sequence
                shifted_sequence_start_value = working_phase_data.loc[working_phase_data.index == sequence_in_data[0]][sensor].iloc[0] - diff
                end_of_sequence_value = working_phase_data.loc[working_phase_data.index == sequence_in_data[-1]][sensor].iloc[0]

                new_sequence_values = DataUtilityClass.scale_list_of_values(working_phase_data.loc[sequence_mask][sensor].values,
                                                                                        shifted_sequence_start_value,
                                                                                        end_of_sequence_value)
                # logging.debug(f"old values: {working_phase_data.loc[sequence_mask][sensor].values}")
                # logging.debug(f"new values: {new_sequence_values}")
                # with a positive gradient, the sequence start value is lower than the sequence end value
                working_phase_data.loc[sequence_mask, sensor] = new_sequence_values
                # logging.debug("new values in data: {}".format(working_phase_data.loc[sequence_mask][sensor].values))
                # logging.debug("By-Gradient Aligner: Used positive gradient with sequence length: {}".format(len(sequence_in_data)))
                self.selected_sequence_length = len(sequence_in_data)
                self.selected_gradient= 'positive'
                used_sequence = True
                break
            elif 'negative':
                # here we shift in the direction of the gradient, so we need to be careful
                # if diff > seq start - seq end, we need to find a new sequence
                sequence_start_value = working_phase_data.loc[working_phase_data.index == sequence_in_data[0]][sensor].iloc[0]
                sequence_end_value = working_phase_data.loc[working_phase_data.index == sequence_in_data[-1]][sensor].iloc[0]

                # logging.debug("diff: {}, sequence start: {}, phase start: {}".format(diff, 
                #                                                                      working_phase_data.loc[working_phase_data.index == sequence_in_data[0]]['seconds'], 
                #                                                                      working_phase_data.loc[working_phase_data.index == new_phase_start_index]['seconds']))
                 

                if diff > sequence_start_value - sequence_end_value:
                    continue

                # we need to shift the sequence start to the phase start
                # and then rescale the sequence values
                phase_start_mask = working_phase_data.index.isin(range(new_phase_start_index, sequence_in_data[0]))
                working_phase_data.loc[phase_start_mask, sensor] = working_phase_data.loc[phase_start_mask][sensor] - diff

                sequence_mask = working_phase_data.index.isin(sequence_in_data)

                # get the values at the start and end of the sequence
                shifted_sequence_start_value = working_phase_data.loc[working_phase_data.index == sequence_in_data[0]][sensor].iloc[0] - diff
                end_of_sequence_value = working_phase_data.loc[working_phase_data.index == sequence_in_data[-1]][sensor].iloc[0]

                new_sequence_values = DataUtilityClass.scale_list_of_values(working_phase_data.loc[sequence_mask][sensor].values,
                                                                            end_of_sequence_value,
                                                                            shifted_sequence_start_value)
                # logging.debug(f"old values: {working_phase_data.loc[sequence_mask][sensor].values}")
                # logging.debug(f"new values: {new_sequence_values}")
                # with a positive gradient, the sequence start value is lower than the sequence end value
                working_phase_data.loc[sequence_mask, sensor] = new_sequence_values
                #logging.debug("By-Gradient Aligner: Used negative gradient with sequence length: {}".format(len(sequence_in_data)))
                self.selected_sequence_length = len(sequence_in_data)
                self.selected_gradient = 'negative'
                used_sequence = True
                break
                
        if used_sequence == False:
            logging.debug("By-Gradient Aligner: Could not find a suitable sequence")
            return data, False

        if flip_values:
            working_phase_data[sensor] = working_phase_data[sensor] * -1
            self._flip_selected_gradient_string()

        if align_previous_phase:
            working_phase_data = working_phase_data.sort_values(by='seconds', ascending=True)
            working_phase_data = working_phase_data.reset_index(drop=True)
            self._flip_selected_gradient_string()
        
        data.loc[mask,sensor] = working_phase_data[sensor].values

        # logging.debug("By-Gradient Aligner: Original data:")
        # logging.debug(data.loc[mask][sensor])
        # logging.debug("By-Gradient Aligner: Changed data:")
        # logging.debug(working_phase_data[sensor])

        return data, True
        

    def _gradient_sequences_sampling(self, positive_gradients_sequences, negative_gradients_sequences):
        """
            returns a random sequence list permutation of positive and negative gradient sequences, with a bit for positive and negative
        """
        positive_gradients_sequences = [
            (sequence, 'positive') for sequence in positive_gradients_sequences
        ]
        negative_gradients_sequences = [
            (sequence, 'negative') for sequence in negative_gradients_sequences
        ]
        combined = positive_gradients_sequences + negative_gradients_sequences
        
        random_permutation_indicies = np.random.permutation(len(combined))
        random_permutation = [combined[index] for index in random_permutation_indicies]

        #random_permutation = np.random.permutation(positive_gradients_sequences + negative_gradients_sequences)
        #random_permutation = DataUtilityClass.get_random_elements(combined, len(combined))
        
        return random_permutation

    def _flip_selected_gradient_string(self):
        if self.selected_gradient == 'positive':
            return 'negative'
        elif self.selected_gradient == 'negative':
            return 'positive'
        else:
            raise ValueError("By-Gradient Aligner: Chosen gradient not set")



        


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

        return data
