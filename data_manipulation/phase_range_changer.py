import pandas as pd
import numpy as np
from data_management.data_handler import DataHandler

from data_manipulation.utility_class import DataUtilityClass
from data_manipulation.abstract_manipulator import AbstractManipulator
import random
import logging

class PhaseRangeChanger(AbstractManipulator):
    """
    basically takes a list of phases and adjusts the phase range 
    thereby making it much flatter or steeper curves ie increasing the peaks and valleys
    
    This is done without changing the borders

    TODO: probably needs some careful tuning of parameters      
          maybe do one where the borders are also changed and we have automatic adjustment no neighbouring phases?
    
    """

    def __init__(self, data_handler : DataHandler) -> None:
        """
            not yet sure if we need an init or all of these functions can be static
        """
        self.data_handler : DataHandler = data_handler

        self.all_functions = ["simple_gaussian"]

    def __str__(self) -> str:  
        return f"PhaseRangeChanger: {self.function_type}+{self.sub_type}, factor {self.factor}, next/previous alignment {self.align_next_phase}/{self.align_previous_phase}"

    def old__apply_range_change_without_borders(self, data : pd.DataFrame, sensor :str, phase_index_list : list, factor : float) -> pd.DataFrame:
        """
            apply a range change to a set of phases without changing the borders
            this way we do not have to align phases

            this works by centering the phase around the "median" of the start and end values,

            @param
                data : pd.DataFrame -- the data to be changed
                sensor : str -- the sensor to be changed
                phase_index_list : list -- the phases to be changed
                factor : float -- factor to change the range of the phases
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]
        # phase_data_min = phase_data[sensor].min()
        # phase_data_max = phase_data[sensor].max()

        phase_data_start = phase_data[sensor].iloc[0]
        phase_data_end = phase_data[sensor].iloc[-1]

        phase_data_median = (phase_data_start + phase_data_end) / 2

        phase_data_sensor = phase_data[sensor] - phase_data_median
        if phase_data_sensor.min() > 0 or phase_data_sensor.max() < 0:
            logging.debug("PhaseRangeChanger: changing without borders not effective")
            logging.debug()

        phase_data_sensor *= factor

        phase_data_sensor += phase_data_median

        data.loc[mask, sensor] = phase_data_sensor
        return data
    
    def _get_mulitpliers_for_gaussian(self, phase_length) -> str:
        """ 
            the idea is to have a value list that rises quickly to the factor, stays there and then falls quickly to 1

            we do this by using a gaussian function to rise and fall
            and have a middle part that is just the factor
        """

        factor = self.factor

        percentage_for_rise_and_fall = 0.1
        adapt_length = int(phase_length * percentage_for_rise_and_fall)
        #function_type = parameters['type']
        # right now only support gaussian function
        values = []

        x_range = 2
        # get x positions for the gaussian function
        x_points = np.linspace(-x_range, x_range, adapt_length*2)
  
        # define the gaussian function
        # not sure which values are best 
        amplitude = 1
        mu = 0
        sigma = 1
        gaussian_function = lambda x : amplitude * np.exp(-((x-mu)**2)/(2*sigma**2))
        # compute gaussian values and scale between 1 and factor
        values = [gaussian_function(x) for x in x_points]
        values = DataUtilityClass.scale_list_of_values(values, 1, factor)

        if self.sub_type == "simple":
            # add the middle part of just factor
            values = values[0:adapt_length] + [factor] * (phase_length - adapt_length*2) + values[adapt_length:]
        elif self.sub_type == "right_tailing":
            # end remains factor
            values = values[0:adapt_length] + [factor] * (phase_length - adapt_length)
        else: # self.sub_type == "left_tailing":
            # start remains factor
            values = [factor] * (phase_length - adapt_length) + values[adapt_length:]

        return values

    def _get_multiplier_list(self, phase_length : int) -> list:
        """ 
            get a list of multipliers for the phase length
            we have seperate functions for different function types

        """

        if self.function_type == "gaussian":            
            return self._get_mulitpliers_for_gaussian(phase_length)
   
            
    def set_manipulation_parameters(self):
        """
            set the parameters for the range change
            these are:
            function_type : str -- the type of function to be used for the range change
            factor : float -- the max/min factor to be used for the range change
            align_next_phase : bool -- if the next phase should be aligned to the end of the current phase
            align_previous_phase : bool -- if the previous phase should be aligned to the start of the current phase

            
        """
        selected_function = DataUtilityClass.get_one_random_element(self.all_functions)

        if selected_function == "simple_gaussian":
            self.function_type = "gaussian"
            self.align_next_phase = False
            self.align_previous_phase = False

            # we have different subtypes for the gaussian function
            # simple, left_tailing, right_tailing: meaning if the factor starts/returns to 1
            # taling may be the wrong word, as it is not really a tail, but we have the factor at start or end
            # 50% chance for simple, 25% for left_tailing and 25% for right_tailing
            if random.random() < 0.5:
                self.sub_type = "simple"
            elif random.random() < 0.5:
                self.sub_type = "left_tailing"
                self.align_previous_phase = True
            else:
                self.sub_type = "right_tailing"
                self.align_next_phase = True

            self.factor = DataUtilityClass.get_random_float(0.4, 1.6, precision=2, exclude=[0.9,1,1.1])
        
        else:
            # should never be raised 
            raise ValueError("Phase Range Changer, set parameters: function type not supported")



        

    def apply_manipulation(self, data : pd.DataFrame, sensor :str, phase_index_list : list) -> pd.DataFrame:
        """
            apply a range change to a set of phases without changing the borders
            this way we do not have to align phases

            this works by centering the phase around the "median" of the data
            this way peaks and valleys are increased or decreased  

            @param
                data : pd.DataFrame -- the data to be changed
                sensor : str -- the sensor to be changed
                phase_index_list : list -- the phases to be changed
                parameters : dict -- should contain factor and function type to increase/decrease the factor
                    this way we start at 1 and do not change the start and end value
                    TODO: factor should be >0, inverting doesn't seem reasoable rn :thinking:
        """
        mask = self.data_handler.get_mask_for_phases(data, phase_index_list)
        phase_data = data.loc[mask]
        phase_data_min = phase_data[sensor].min()
        phase_data_max = phase_data[sensor].max()

        # center the data, this way we can increase and decrease the peaks and valleys
        phase_data_median = (phase_data_min + phase_data_max) / 2


        phase_data_sensor = phase_data[sensor] - phase_data_median

        # check if the range does anything
        # when using min/max this should never occur
        # if true, the range change is the same as just multiplying the data without centering it
        if phase_data_sensor.min() > 0 or phase_data_sensor.max() < 0:
            logging.debug("PhaseRangeChanger: changing without borders not effective")

        phase_length = len(phase_data_sensor)

        multiplier_list = self._get_multiplier_list(phase_length)

        phase_data_sensor *= multiplier_list

        phase_data_sensor += phase_data_median

        data.loc[mask, sensor] = phase_data_sensor
        return data
    

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