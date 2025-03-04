import pandas as pd
import numpy as np
from data_manipulation.abstract_manipulator import AbstractManipulator
from data_management.data_handler import DataHandler
from data_manipulation.utility_class import DataUtilityClass
import random
import logging
from scipy.fftpack import fft, ifft



class NonStructuralAnomaly(AbstractManipulator):
    """
        This is mainly based on the paper "TSAGen: Synthetic Time Series Generation for KPI Anomaly Detection"
        where structural anomalies are defined as anomalies in the structural properties of the time series (e.g., trend, seasonality, etc.)
        and non-structural occur randomly

        we also use the two shapes/types they defined
        with our implementation as their code had literally 0 comments that described what the code does

        type1 is a spike shape (positive or negative)
            we use the start and end values to fit the functions a and b
            we use a and b from the paper but and then added more parameters
        type2 is a drop/increase with pos/neg gradient and return to normal (either drop/rise or exponential decay/growth)

        we mostly use the same parameters names, so the paper can be used as reference

        this is the original code:
        https://github.com/AprilCal/TSAGen/blob/master/generator/pattern.py

    """

    def __init__(self, data_handler : DataHandler):
        super().__init__(data_handler)
        self.on_selected_phases = True
        self.on_selected_sensors = True
        self.anomaly_types = ["type1", "type2"]

        self.type_1_subtypes_initial = ["up", "down"]
        self.type_1_invert_function_b = ["normal"]
    
    def __str__(self):
        if self.anomaly_type == "type1":
            string_type = f" {self.anomaly_type}+{self.sub_type} | factor {round(self.height_factor,2)} | flip {self.flip_values}"
        elif self.anomaly_type == "type2":
            string_type = f" {self.anomaly_type} h1/h2 {round(self.h1_factor,2)} - {round(self.h2_factor,2)}"
        else:
            raise ValueError(f"Anomaly type {self.anomaly_type} not supported")
        return f"NonStructuralAnomaly | {string_type} | length {self.anomaly_length}"

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            applies the manipulation to the data
            depends on the anomaly type
            later adds noise to the inserted anomaly
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list).copy()
        phase_data = data.loc[mask]
        phase_length = len(phase_data)
        original_sensor_data = data[sensor].copy()


        # first goal: get anomaly mask
        # so determine the start and end of the anomaly in the data
        start_index = phase_data.index[0]
        end_index = phase_data.index[-1]

        self.anomaly_length = int(phase_length * self.w) # w is the widths factor

        anomaly_start = random.randint(0, phase_length - self.anomaly_length) #  start before phase length - anomaly length
        self.anomaly_start_index = start_index + anomaly_start
        self.anomaly_end_index = self.anomaly_start_index + self.anomaly_length # exclusive

        self.anomaly_start_value = data.loc[self.anomaly_start_index][sensor]
        self.anomaly_end_value = data.loc[self.anomaly_end_index][sensor]

        self.data_midrange = data[sensor].max() - data[sensor].min()

        assert self.anomaly_end_index <= end_index
        
        # now we have the start and end index of the anomaly
        anomaly_mask = (data.index >= self.anomaly_start_index) & (data.index < self.anomaly_end_index)

        # assert that the anomaly mask has the correct length
        assert self.anomaly_length == data.loc[anomaly_mask].shape[0], f"{self.anomaly_length} != {data.loc[anomaly_mask].shape[0]}"

        if self.anomaly_type == "type1":
            data = self._apply_type_1_anomaly(data, anomaly_mask, sensor)
        elif self.anomaly_type == "type2":
            self._apply_type_2_anomaly(data, anomaly_mask, sensor)
        else:
            raise ValueError(f"Anomaly type {self.anomaly_type} not supported")
        
        #data = self._insert_noise_with_mask(data, original_sensor_data, anomaly_mask, sensor)

        return data
    

    def _apply_type_1_anomaly(self, data : pd.DataFrame, anomaly_mask, sensor : str) -> pd.DataFrame:
        """
            apply a type 1 anomaly to the data
            the anomaly is a spike shape
            we use the start and end values to fit the functions a and b
            use start < end as base case and flip the data otherwise

            the functions a and b have seperate definitions
            to understand a and b and the impact of the parameters, it is helpful to plot them using geogebra for example
            these are independent of the data index ranges so start at 0 
        """

        anomaly_start_value = self.anomaly_start_value
        anomaly_end_value = self.anomaly_end_value

        # flip the values to reduce cases (similar to PhaseRangeChanger)
        flip_values = False
        if anomaly_start_value > anomaly_end_value:
            data.loc[anomaly_mask, sensor] *= -1
            anomaly_start_value *= -1
            anomaly_end_value *= -1
            flip_values = True
        # now start < end

        diff_start_end_value = anomaly_end_value - anomaly_start_value

        direction, invert_function_b = self.sub_type.split("+")

        # for up spikes, we have to increase above the end value
        # so we simply multiply the end value with the height factor
        if direction == "up":
            spike_height = (self.data_midrange + diff_start_end_value) * self.height_factor
        # for down it doesnt matter, we just have to return to end value
        elif direction == "down":
            # multiply by -1 to get the negative spike
            spike_height = -(self.data_midrange) * self.height_factor
        # the +/- anomaly_start/end_value is to correct for the up/down movement of the function curve according to the start/end values

        if flip_values:
            spike_height *= -1

        # goal: masks for section a and b in the original data
        # split the analysis length into two sections based on e1 (and e2)        
        length_of_section_a = int(self.e1_factor * self.anomaly_length)
        length_of_section_b = self.anomaly_length - length_of_section_a

        # determine x values for the a and b functions
        # these are independent of the data index ranges
        section_a_x_values = np.arange(0, length_of_section_a)
        section_b_x_values = np.arange(length_of_section_a, self.anomaly_length)

        # get the index mask for the sections based on the length of section a
        section_a_mask = (data.index >= self.anomaly_start_index) & (data.index < self.anomaly_start_index + length_of_section_a)
        section_b_mask = (data.index >= self.anomaly_start_index + length_of_section_a) & (data.index < self.anomaly_end_index)

        # get functions, theese are depednent on the data index ranges
        function_a = self._function_a_for_type1(spike_height, anomaly_start_value, length_of_section_a)
        function_b = self._function_b_for_type2(spike_height, diff_start_end_value, anomaly_end_value, length_of_section_a, length_of_section_b)
        

        assert len(section_a_x_values) + len(section_b_x_values) == data.loc[anomaly_mask].shape[0], f"{len(section_a_x_values)} + {len(section_b_x_values)} != {data.loc[anomaly_mask].shape[0]}"

        # apply the functions to the data
        data.loc[section_a_mask, sensor] += function_a(section_a_x_values)
        # we invert b if set, so we get a up to height, down to -height and return to end value
        if invert_function_b == "invert":
            data.loc[section_b_mask, sensor] -= function_b(section_b_x_values)
        elif invert_function_b == "normal":
            data.loc[section_b_mask, sensor] += function_b(section_b_x_values)
        else:
            raise ValueError(f"Unknown function inversion {invert_function_b}")
        

        if flip_values:
            data.loc[anomaly_mask, sensor] *= -1

        self.flip_values = flip_values

        return data

    def _function_a_for_type1(self, h, start_value, anomaly_e1_length):
        return lambda x : h*(np.e**((-np.log(1/1000)/anomaly_e1_length)*(x-anomaly_e1_length)))

    def _function_b_for_type2(self, h, diff, end_value, anomaly_e1_length, anomaly_e2_length):
        return lambda x : (h) *(np.e**((np.log(1/1000)/anomaly_e2_length)*(x-anomaly_e1_length))) 
        

    def _apply_type_2_anomaly(self, data : pd.DataFrame, anomaly_mask, sensor : str) -> pd.DataFrame:
        """
            apply a type 2 anomaly to the data
            @param
            data: the data frame
            anomaly_mask: the mask for the anomaly
            sensor: the sensor to apply the anomaly to
        """
        def flip_height(height, positive):
            if positive:
                return height
            else:
                return -height

        h1_height = self.h1_factor *  self.data_midrange
        h2_height = self.h2_factor *  self.data_midrange

        h1_height = flip_height(h1_height, self.h1_positive)
        h2_height = flip_height(h2_height, self.h2_positive)

        anomaly_values = np.linspace(h1_height, h2_height, self.anomaly_length)

        data.loc[anomaly_mask, sensor] += anomaly_values
        return 

    def _get_difference_to_factor_multiply(self, factor, value):
        return abs(factor * value)- abs(value)

    def set_manipulation_parameters(self, align_to_next : bool = None, align_to_previous : bool = None):
        """
            sets the internal parameters for the anomaly injector

            the placement of the anomaly is later determinent by the sensor data and the phase_index_list
        """
        self.w = DataUtilityClass.get_random_float(0.07, 0.17, 2)

        self.anomaly_type = DataUtilityClass.get_one_random_element(self.anomaly_types)

        if self.anomaly_type == "type1": # type1
            self.height_factor = self._get_random_anomaly_height()

            # we need:
            # e1 and e2 do split the width for a and b function
            # height h
            # up or down spike
            # width w as percentage of the phase length

            # dont have the middle point between a and b functions too close to the border
            mid_point_border_percentage = 0.05
            self.e1_factor = DataUtilityClass.get_random_float(mid_point_border_percentage, 
                                                        1-mid_point_border_percentage, 
                                                        2)
            self.e2_factor = 1 - self.e1_factor

            # indicates if function a goes up or down
            direction = DataUtilityClass.get_one_random_element(self.type_1_subtypes_initial)
            # indicates if function b is inverted or not, ie starts oppsite of a
            invert = DataUtilityClass.get_one_random_element(self.type_1_invert_function_b)
            self.sub_type = f"{direction}+{invert}"


        elif self.anomaly_type == "type2":
            self.h1_factor = self._get_random_anomaly_height()
            self.h2_factor = self._get_random_anomaly_height()
            self.h1_positive = DataUtilityClass.get_random_bool()
            self.h2_positive = DataUtilityClass.get_random_bool()

            
        else:
            raise ValueError(f"Anomaly type {self.anomaly_type} not supported")
        
    def _get_random_anomaly_height(self):
        """
            return random anomaly height, needed 3 times, so easier to change here
        """
        return DataUtilityClass.get_random_float(0.07, 0.3, 2)

    def _insert_noise_with_mask(self, data : pd.DataFrame, original_sensor_data : pd.Series, data_section_mask, sensor : str) -> pd.DataFrame:
        """
            Currently obsolete, went with additive instead of replacing data, which preserves data noise


            inserts noise into the data
            we use a simple rolling mean to get the trend and then subtract it from the original data
            we need the original sensor data, since the added anomaly has no noise and is already in the data frame

            TODO: experiment more with fft to get the trend
        """
        # TODO: think of some way to scale the windows length based on the data
        window = 10  
        trend = original_sensor_data.rolling(window=window).mean().values      
        data['sensor_noise'] =( original_sensor_data - trend) * 3
        # factor 3 to increase the noise, the anomaly looked to clean
        # currently noise is very low anyway


        # fft_values = fft(sensor_data.values)
        # frequencies = np.fft.fftfreq(len(fft_values))

        # # Filter frequencies to remove low-frequency trend
        # threshold = 0.2  # Define threshold for filtering
        # filtered_fft = fft_values.copy()
        # filtered_fft[np.abs(frequencies) < threshold] = 0
        # Reconstruct noise using inverse FFT
        # add noise column to to able to use the data_section_mask
        #data.loc['noise'] = np.real(ifft(filtered_fft))

        #wadaw
        # Add noise to the data
        data.loc[data_section_mask, sensor] += data.loc[data_section_mask, 'sensor_noise']
        #delete noise
        #data.drop(columns=['sensor_noise'], inplace=True)
        return data