import pandas as pd
import random

import logging
logging.basicConfig(level=logging.DEBUG,
                        #format="%(asctime)s [%(levelname)s] %(message)s",
                        format = '%(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

import options as opt
from data_management.data_loader import DataLoader
from data_management.data_handler import DataHandler

from data_manipulation.utility_class import DataUtilityClass

from data_visualizer import DataVisualizer


from data_manipulation.abstract_manipulator import AbstractManipulator
from data_manipulation.phase_function_anomaly import PhaseFunctionAnomaly
from data_manipulation.by_gradient_aligner import ByGradientAligner
from data_manipulation.phase_range_changer import PhaseRangeChanger


class DatasetGenerator():
    """
        This class is responsible for generating a dataset with anomalies
        It uses the DataHandler to get the data and AbstracManipulator subclasses to inject anomalies and align phases
        The DataVisualizer is used to visualize the data

        TODO: save method

    """

    def __init__(self, number_of_anomaly_samples : int, max_number_of_sensors : int) -> None:
        """
            Initialize the DatasetGenerator
            set up the data handler, the anomaly injectors, the aligners and visualizer

        """
        # ------------------------------------------------
        # Injection Setting
        # TODO: include severity parameter
        # ------------------------------------------------

        self.number_of_anomaly_samples = number_of_anomaly_samples

        # TODO: support more than 1
        max_number_of_sensors = max_number_of_sensors

        #------------------------------------------------
        # Initilization and Setup
        #------------------------------------------------

        self.data_handler = DataHandler()
        self.sensor_list = opt.MOST_IMPORTANT_SENSOR_COLUMNS

        self.phase_index_list = self.data_handler.get_phase_indices_list()

        # set seed
        random.seed(42)


        self.phase_function_anomaly_injector = PhaseFunctionAnomaly(self.data_handler)
        self.by_gradient_aligner = ByGradientAligner(self.data_handler)
        phase_range_changer = PhaseRangeChanger(self.data_handler)

        # anomaly_injectors = [
        #     "phase_function_injector" : phase_function_anomaly_injector,
        #     #"phase_range_changer" : phase_range_changer
        # ]

        # store all anomaly injectors in a list
        self.anomaly_injector_list = [
            self.phase_function_anomaly_injector
        ]


        # store all aligners in a list
        # except the by_gradient_aligner, since we currently use this as preferred method
        self.aligner_list = [
            self.phase_function_anomaly_injector
        ]

        visual_rows = 3 # for original, injected and aligned data
        visual_columns = self.number_of_anomaly_samples // visual_rows + 1

        self.data_visualizer = DataVisualizer((visual_rows, number_of_anomaly_samples))

        logging.INFO("Setup complete")


    def generate_dataset(self):
        """
            generates a dataset with anomalies
        """
        
        #------------------------------------------------
        # Big loop 
        #------------------------------------------------

        for injected_sample_iterator in range(self.number_of_anomaly_samples):

            # get random data sample (samples are groups)
            group_index = random.randint(0, self.data_handler.get_number_of_groups())
            data_sample = self.data_handler.get_group_by_index(group_index)
            logging.info("Selected group index: {}".format(group_index))

            # TODO: Support multiple anomalies, such that we loop starting from here

            # get random phase slice as list
            phase_index_list = DataUtilityClass.get_random_slice(phase_index_list)
            logging.info("Selected phase indices: {}".format(phase_index_list))

            # select sensor
            # TODO: support multiple sensors
            selected_sensor = DataUtilityClass.get_random_elements(self.sensor_list, max_number_of_elements = self.max_number_of_sensors)
            # returns list, so we take the first element
            selected_sensor = selected_sensor[0]
            logging.info("Selected sensor: {}".format(selected_sensor))

            # plot initial data
            self.data_visualizer.plot_at_grid_position(grid_position=(0,injected_sample_iterator),
                                                    data=data_sample,
                                                    sensor_list=selected_sensor,
                                                    add_phase_lines=True,
                                                    title = "Initial Data")

            # select injector
            # get parameters
            # inject anomaly
            # check if aligning is necessary
            # try by gradient
            # else use function anomaly to align

            #------------------------------------------------
            # Inject Anomaly
            #------------------------------------------------

            # select injector
            selected_anomaly_injector = DataUtilityClass.get_one_random_element(self.anomaly_injector_list)

            assert isinstance(selected_anomaly_injector, AbstractManipulator)

            # set internal parameters
            # TODO: add severity parameter
            selected_anomaly_injector.set_manipulation_parameters()

            # apply
            data_sample = selected_anomaly_injector.apply_manipulation(data = data_sample,
                                                                    sensor = selected_sensor,
                                                                    phase_index_list = phase_index_list)
            
            # plot injected data
            self.data_visualizer.plot_at_grid_position(grid_position=(1,injected_sample_iterator),
                                                    data=data_sample,
                                                    sensor_list=selected_sensor,
                                                    add_phase_lines=True,
                                                    title="Injected Data")
            
            #------------------------------------------------
            # Align Anomaly
            #------------------------------------------------


            self.handle_alignment(data = data_sample,
                                sensor = selected_sensor,
                                phase_index_list = phase_index_list,
                                align_to_next = selected_anomaly_injector.requires_alignment_of_next_phase(),
                                align_to_previous = selected_anomaly_injector.requires_alignment_of_previous_phase())            
            
            # if selected_anomaly_injector.requires_alignment_of_next_phase():
            #     self.handle_alignment(align_to_next = True, align_to_previous = False)
            # if selected_anomaly_injector.requires_alignment_of_previous_phase():
            #     self.handle_alignment(align_to_next = False, align_to_previous = True)

        

            # plot aligned data
            self.data_visualizer.plot_at_grid_position(grid_position=(2,injected_sample_iterator),
                                                    data=data_sample,
                                                    sensor_list=selected_sensor,
                                                    add_phase_lines=True,
                                                    title="Aligned Data")

        # TODO: save data



    def handle_alignment(self,
                         data_sample : pd.DataFrame,
                         selected_sensor : str,
                         phase_index_list : list,
                         align_to_next : bool, 
                         align_to_previous : bool):
        """
            handles the alignent of phases, we try to align by gradient first, if this fails we use other aligners
            method is the same as anomaly injection, just with the align bool parameters

            the handling of next and previous is the same, so maybe I can simplfy this
            
        """

        if align_to_next:        
            data_sample, success = self.by_gradient_aligner.align_by_gradient(data = data_sample, 
                                                                              sensor = selected_sensor, 
                                                                              phase_index_list = phase_index_list,
                                                                              align_to_next = align_to_next)
            if not success:
                selected_aligner = DataUtilityClass.get_one_random_element(self.aligner_list)
                assert isinstance(selected_aligner, AbstractManipulator)

                selected_aligner.set_manipulation_parameters(align_to_next = align_to_next)
                data_sample = selected_aligner.apply_manipulation(data = data_sample,
                                                                  sensor = selected_sensor,
                                                                  phase_index_list = phase_index_list, 
                                                                  align_to_next = align_to_next)
                
        if align_to_previous:
            data_sample, success = self.by_gradient_aligner.align_by_gradient(data = data_sample, 
                                                                              sensor = selected_sensor, 
                                                                              phase_index_list = phase_index_list,
                                                                              align_to_previous = align_to_previous)
            if not success:

                selected_aligner = DataUtilityClass.get_one_random_element(self.aligner_list)
                assert isinstance(selected_aligner, AbstractManipulator)

                selected_aligner.set_manipulation_parameters(align_to_previous = align_to_previous)
                data_sample = selected_aligner.apply_manipulation(data = data_sample,
                                                                  sensor = selected_sensor,
                                                                  phase_index_list = phase_index_list, 
                                                                  align_to_previous = align_to_previous)