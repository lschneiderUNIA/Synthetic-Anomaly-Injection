import pandas as pd
import random

import logging
logging.basicConfig(level=logging.DEBUG,
                        #format="%(asctime)s [%(levelname)s] %(message)s",
                        format = '%(levelname)s - %(message)s',
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING) 

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
        self.max_number_of_sensors = max_number_of_sensors

        #------------------------------------------------
        # Initilization and Setup
        #------------------------------------------------

        self.data_handler = DataHandler()
        self.sensor_list = opt.MOST_IMPORTANT_SENSOR_COLUMNS
        logging.info("Sensor list: {}".format(self.sensor_list))

        self.phase_index_list = self.data_handler.get_phase_indices_list()

        # set seed
        random.seed(7767)


        phase_function_anomaly_injector = PhaseFunctionAnomaly(self.data_handler)
        phase_range_changer = PhaseRangeChanger(self.data_handler)

        self.by_gradient_aligner = ByGradientAligner(self.data_handler)

        # anomaly_injectors = [
        #     "phase_function_injector" : phase_function_anomaly_injector,
        #     #"phase_range_changer" : phase_range_changer
        # ]

        # store all anomaly injectors in a list
        self.anomaly_injector_list = [
            phase_function_anomaly_injector,
            phase_range_changer
        ]
        logging.debug("Anomaly injectors: {}".format(self.anomaly_injector_list))


        # store all aligners in a list
        # except the by_gradient_aligner, since we currently use this as preferred method
        # but does not work for all cases
        # TODO: if I add more aligners, we could just try a random one until one is successful
        self.aligner_list = [
            phase_function_anomaly_injector
        ]

        logging.debug("Aligners: {}".format(self.aligner_list))

        rows = self.number_of_anomaly_samples
        columns = 1

        self.data_visualizer = DataVisualizer((rows, columns))

        logging.info("Setup complete")


    def generate_dataset(self):
        """
            generates a dataset with anomalies
        """
        
        #------------------------------------------------
        # Big loop 
        #------------------------------------------------

        for self.injected_sample_iterator in range(self.number_of_anomaly_samples):
            logging.info("Generating sample: {}".format(self.injected_sample_iterator))

            # get random data sample (samples are groups)
            group_index = random.randint(0, self.data_handler.get_number_of_groups())
            data_sample = self.data_handler.get_group_by_index(group_index)
            logging.debug("Selected group index: {}".format(group_index))

            # TODO: Support multiple anomalies, such that we loop starting from here

            # get random phase slice as list
            selected_phases = DataUtilityClass.get_random_slice(self.phase_index_list)
            logging.debug("Selected phase indices: {}".format(selected_phases))

            # select sensor
            # TODO: support multiple sensors
            selected_sensor = DataUtilityClass.get_random_elements(self.sensor_list,
                                                                   max_number_of_elements = self.max_number_of_sensors)
            # returns list, so we take the first element
            selected_sensor = selected_sensor[0]
            logging.debug("Selected sensor: {}".format(selected_sensor))

            # plot initial data
            self.data_visualizer.plot_at_grid_position(grid_position=(self.injected_sample_iterator,0),
                                                    data=data_sample,
                                                    x_column='seconds',
                                                    y_column=selected_sensor,
                                                    plot_color='blue',
                                                    x_limits=(0, None),
                                                    y_limits=(0, None),
                                                    plot_label="Original Data",
                                                    add_phase_lines=True)

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
                                                                    phase_index_list = selected_phases)
            
            logging.debug(f"Inject anomaly: {selected_anomaly_injector}")
            
            #plot injected data
            # self.data_visualizer.plot_at_grid_position(grid_position=(self.injected_sample_iterator,0),
            #                                         data=data_sample,
            #                                         x_column='seconds',
            #                                         y_column=selected_sensor,
            #                                         plot_color='green',
            #                                         add_phase_lines=True)
            
            #------------------------------------------------
            # Align Anomaly
            #------------------------------------------------

            # check if alignment is necessary
            align_next_phase = selected_anomaly_injector.requires_alignment_of_next_phase()
            align_previous_phase = selected_anomaly_injector.requires_alignment_of_previous_phase()
            # we handle the selected phase edges in here, so we don't have to do it in every aligner subclass
            if selected_phases[0] ==  self.phase_index_list[0]:
                align_previous_phase = False
            if selected_phases[-1] ==  self.phase_index_list[-1]:
                align_next_phase = False

            logging.debug("Align to next/previous: {}/{}".format(align_next_phase, align_previous_phase))

            data_sample = self.handle_alignment(data_sample = data_sample,
                                selected_sensor = selected_sensor,
                                selected_phases = selected_phases,
                                align_next_phase = align_next_phase,
                                align_previous_phase = align_previous_phase)       


            title = str(selected_anomaly_injector) + " | " + str(selected_sensor) + " | " + str(selected_phases)   

            # plot final data
            self.data_visualizer.plot_at_grid_position(grid_position=(self.injected_sample_iterator, 0),
                                                    data=data_sample,
                                                    x_column='seconds',
                                                    y_column=selected_sensor,
                                                    plot_color='red',
                                                    x_limits=(0, None),
                                                    y_limits=(0, None),
                                                    plot_label="Injected Data",
                                                    add_phase_lines=True)
            
            self.data_visualizer.set_title_at_position(grid_position=(self.injected_sample_iterator, 0), title=title)
            

        # TODO: save data

        self.data_visualizer.show_data()

    def handle_alignment(self,
                         data_sample : pd.DataFrame,
                         selected_sensor : str,
                         selected_phases : list,
                         align_next_phase : bool, 
                         align_previous_phase : bool) -> pd.DataFrame:
        """
            handles the alignent of phases, we try to align by gradient first, if this fails we use other aligners
            method is the same as anomaly injection, just with the align bool parameters

            the handling of next and previous is the same, so maybe I can simplfy this
            
        """

        if align_next_phase:        
            new_selected_phases = [selected_phases[-1]+1] # compute this here, so its not handeled by each subclass
            logging.debug("New selected phases: {}".format(new_selected_phases))
            assert all([phase in self.phase_index_list] for phase in new_selected_phases)
            # TODO: support more than 1 phase during alignment
            data_sample, success = self.by_gradient_aligner.align_by_gradient(data = data_sample, 
                                                                              sensor = selected_sensor, 
                                                                              phase_index_list = new_selected_phases,
                                                                              align_next_phase = True,
                                                                              align_previous_phase = False)
            
            if not success:
                logging.debug("Alignment by gradient failed, trying other aligners")

                selected_aligner = DataUtilityClass.get_one_random_element(self.aligner_list)
                assert isinstance(selected_aligner, AbstractManipulator)

                selected_aligner.set_manipulation_parameters(align_next_phase = True)
                data_sample = selected_aligner.apply_manipulation(data = data_sample,
                                                                  sensor = selected_sensor,
                                                                  phase_index_list = new_selected_phases)
                logging.debug(f"Align next: {selected_aligner}")

                
        if align_previous_phase:
            new_selected_phases = [selected_phases[0]-1]
            logging.debug("New selected phases: {}".format(new_selected_phases))

            assert all([phase in self.phase_index_list] for phase in new_selected_phases)

            data_sample, success = self.by_gradient_aligner.align_by_gradient(data = data_sample, 
                                                                              sensor = selected_sensor, 
                                                                              phase_index_list = new_selected_phases,
                                                                              align_previous_phase = True,
                                                                              align_next_phase = False)
            
            if not success:
                logging.debug("Alignment by gradient failed, trying other aligners")
                selected_aligner = DataUtilityClass.get_one_random_element(self.aligner_list)
                assert isinstance(selected_aligner, AbstractManipulator)

                selected_aligner.set_manipulation_parameters(align_previous_phase = True)
                data_sample = selected_aligner.apply_manipulation(data = data_sample,
                                                                  sensor = selected_sensor,
                                                                  phase_index_list = new_selected_phases)
                logging.debug(f"Align previous: {selected_aligner}")

        
        return data_sample


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(number_of_anomaly_samples = 3, max_number_of_sensors = 1)
    dataset_generator.generate_dataset()