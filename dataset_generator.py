import pandas as pd
import random
import numpy as np
import logging
import json
import datetime

# set up logging
logging.basicConfig(level=logging.DEBUG,
                        #format="%(asctime)s [%(levelname)s] %(message)s",
                        format = '%(levelname)s - %(message)s',
)
# surpress matplotlib and PIL warnings in logger
# if we turn on debug, it just gets spammed
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING) 

logger = logging.getLogger(__name__)

import options_rational as op
from data_management.data_loader import DataLoader
from data_management.data_handler import DataHandler
from data_visualizer import DataVisualizer
from data_manipulation.utility_class import DataUtilityClass

from data_manipulation.abstract_manipulator import AbstractManipulator
from data_manipulation.phase_function_anomaly import PhaseFunctionAnomaly
from data_manipulation.by_gradient_aligner import ByGradientAligner
from data_manipulation.phase_range_changer import PhaseRangeChanger
from data_manipulation.non_structural_anomaly import NonStructuralAnomaly
from data_manipulation.phase_length_changer import PhaseLengthChanger
from data_manipulation.point_anomaly import PointAnomalyInserter
from data_manipulation.data_dropout import DataSectionDropout
from data_manipulation.noise_insertion import NoiseInsertion

from metrics.metric_calculations import MetricCalculations

class DatasetGenerator():
    """
        This class is responsible for generating a dataset with anomalies
        It uses the DataHandler to get the data and AbstracManipulator subclasses to inject anomalies and align phases
        The DataVisualizer is used to visualize the data

        TODO: save method

    """

    def __init__(self, 
                 number_of_anomaly_samples : int, 
                 max_number_of_sensors : int, 
                 save_as_pdf : bool = False,
                 save_file : bool = False,
                 filename : str = None) -> None:
        """
            Initialize the DatasetGenerator
            set up the data handler, the anomaly injectors, the aligners and visualizer

        """


        # ------------------------------------------------
        # Injection Setting
        # TODO: include severity parameter
        # ------------------------------------------------

        self.number_of_anomaly_samples = number_of_anomaly_samples

        # support for multiple sensors, more based on correlated sensor data
        self.max_number_of_sensors = max_number_of_sensors

        #------------------------------------------------
        # Initilization
        #------------------------------------------------

        self.data_handler = DataHandler()
        self.sensor_list = self.data_handler.get_data_sensor_list()
        logging.info("Sensor list: {}".format(self.sensor_list))

        self.phase_index_list = self.data_handler.get_phase_indices_list()

        seed =14
        random.seed(seed)
        np.random.seed(seed)

        # ANOMALY INJECTORS
        phase_function_anomaly_injector = PhaseFunctionAnomaly(self.data_handler)
        phase_range_changer = PhaseRangeChanger(self.data_handler)
        non_structural_anomaly_injector = NonStructuralAnomaly(self.data_handler)
        phase_length_changer = PhaseLengthChanger(self.data_handler)
        point_anomaly_inserter = PointAnomalyInserter(self.data_handler)  
        data_dropout_inserter = DataSectionDropout(self.data_handler)  
        noise_inserter = NoiseInsertion(self.data_handler)

        self.anomaly_injector_list = [
            phase_function_anomaly_injector,
            phase_range_changer,
            non_structural_anomaly_injector,
            phase_length_changer,
            point_anomaly_inserter,
            data_dropout_inserter,
            noise_inserter
        ]
        logging.debug("Anomaly injectors: {}".format([type(inj).__name__ for inj in self.anomaly_injector_list]))

        # ALIGNERS
        self.by_gradient_aligner = ByGradientAligner(self.data_handler)
        """
            we dont add the gradient aligner, since rn we use it as the preferred method for alignment
            phase_function is the fallback
            This is an area for improvements
        """
        self.aligner_list = [
            phase_function_anomaly_injector
            # phase_range_changer? simply scale values to new min max in phase (portion) with utility func
        ]
        # TODO: if I add more aligners, we could just try a random one until one is successful

        logging.debug("Aligners: {}".format([type(aligner).__name__ for aligner in self.aligner_list]))

        # ------------------------------------------------
        # SETUP
        # ------------------------------------------------

        self.save_as_pdf = save_as_pdf
        self.save_file = save_file
        if save_file and filename is not None:
           timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           self.filename = f"{filename}_{timestamp}"
        if save_file and filename is None:
            raise ValueError("If save_file is True, filename must be set")


        rows = self.number_of_anomaly_samples
        columns = max_number_of_sensors
        self.data_visualizer = DataVisualizer((rows, columns), 
                                              save_as_pdf_bool = self.save_as_pdf,
                                              filename = self.filename)

        logging.info("Setup complete")

    

    def generate_dataset(self):
        """
            generates a dataframe of samples
            can save them as pdf or file
        """
        new_datasamples_list = []   
        statistics_for_dataset = {}
        #------------------------------------------------
        # Big loop 
        #------------------------------------------------

        for self.injected_sample_iterator in range(self.number_of_anomaly_samples):
            logging.info("Generating sample: {}".format(self.injected_sample_iterator))

            data_sample, group_index, injection_information = self.generate_sample()          

            #------------------------------------------------
            # METRICS
            #------------------------------------------------
            original_sample = self.data_handler.get_group_by_index(group_index)
            mape_function = MetricCalculations.calc_mape
            euclidean_function = MetricCalculations.calc_euclidean
            dtw_function = MetricCalculations.calculate_dtw
            fast_dtw_function = MetricCalculations.calculate_fast_dtw

            # MAPE metric
            mape = MetricCalculations.compute_metric_for_dataframes(original_sample, data_sample, self.sensor_list, mape_function)

            logging.debug("MAPE: {}".format(mape))
            clean_mape = MetricCalculations.clean_metric_dictionary(mape)
            self.title += f"\n MAPE: {clean_mape} "
            injection_information["mape"] = clean_mape

            # # DTW Metric
            # dtw = MetricCalculations.compute_metric_for_dataframes(original_sample, data_sample, self.sensor_list, dtw_function)
            # logging.debug("DTW: {}".format(dtw))
            # self.title += f"\n DTW: {dtw} "


            #------------------------------------------------
            # SAVING, TITLE, ETC
            #------------------------------------------------
            
            self.data_visualizer.set_title_at_position(grid_position=(self.injected_sample_iterator, 0), title=self.title)
            
            if self.save_as_pdf:
                self.data_visualizer.save_as_pdf()
            if self.save_file:
                new_datasamples_list.append(data_sample)
                group_keys = injection_information["group_keys"]
                statistics_for_dataset[str(group_keys)] = injection_information

                
        #------------------------------------------------
        # FINAL SAVING ETC
        #------------------------------------------------

        directory_name = op.GENERATED_DATA_DIRECTORY

        if not self.save_as_pdf:
            self.data_visualizer.show_data()
        else: 
            self.data_visualizer.close_pdf()
        if self.save_file:
            new_dataset = pd.concat(new_datasamples_list, ignore_index=True)
            new_dataset.to_parquet(f"{directory_name}/{self.filename}.parquet", index = True)
            logging.info("Dataset saved as parquet: {}".format(f"{directory_name}/{self.filename}.parquet"))
            # save statistics as json
            with open(f"{directory_name}/{self.filename}_statistics.json", "w") as fp:
                json.dump(statistics_for_dataset, fp, indent=4)

            

            

        logging.info("Dataset generation complete")
        


    def generate_sample(self):
        """
            generates a single sample, can be used on its own
            but is mainly used by the generate dataset method

            @param

            returns: data_sample, group_index  
        """
        # get random data sample (samples are groups)
        # TODO: move this to get_random_group, currently useful to be able to set group index
        group_index = random.randint(0, self.data_handler.get_number_of_groups())
        #group_index = 296
        data_sample = self.data_handler.get_group_by_index(group_index)
        logging.debug("Selected group index: {}".format(group_index))

        # TODO: possibly support multiple anomalies, such that we loop starting from here, Lukas doesnt feel its necessary
        # also keeps anomaly types singular, makes analysis easier

        # get random phase slice as list
        selected_phases = DataUtilityClass.get_random_slice(self.phase_index_list)
        #selected_phases = [2]
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
        
        #logging.debug(f"Inject anomaly: {selected_anomaly_injector}")

        # keys of group
        group_keys = self.data_handler.get_keys_of_group_index(group_index)
        group_keys = [str(key) for key in group_keys]
        group_keys_as_string = str(group_keys)

        self.title = f"Group index: {group_index} | {group_keys_as_string} \n"

        self.title += self._get_injector_title(selected_anomaly_injector, selected_phases, selected_sensor)

        logging.debug(f"Injection: {self.title}")
        
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

        # plot final data
        self.data_visualizer.plot_at_grid_position(grid_position=(self.injected_sample_iterator, 0),
                                                data=data_sample,
                                                x_column='seconds', 
                                                y_column= selected_sensor,
                                                plot_color='red',
                                                x_limits=(0, None),
                                                y_limits=(0, None),
                                                plot_label="Injected Data",
                                                add_phase_lines=True)#
        
        injection_information = {
            "group_index": group_index,
            "group_keys": group_keys,
            "selected_phases": selected_phases,
            "selected_sensor": selected_sensor,
            "anomaly_injector": selected_anomaly_injector.__class__.__name__,
            "anomaly_injector_complete_info": str(selected_anomaly_injector),
            "align_next_phase": align_next_phase,
            "align_previous_phase": align_previous_phase
        }
        
        return data_sample, group_index, injection_information
    

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
            else: 
                selected_aligner = self.by_gradient_aligner
            self.title += f"\n Next phase: {selected_aligner}"                

                
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
            else:
                selected_aligner = self.by_gradient_aligner
            self.title += f"\n Previous phase: {selected_aligner}"
        
        return data_sample
    
    def _plot_gradient_sequences(self, data_sample : pd.DataFrame, selected_sensor : str):
        """
            used this during bug fixing to plot the gradient sequences
        """
        for phase_index in self.phase_index_list:

                mask = self.data_handler.get_mask_for_phases(data_sample, [phase_index])
                phase_data = data_sample[mask]

                positive, negative = self.by_gradient_aligner.find_gradient_sequence(data = phase_data,
                                                                            sensor = selected_sensor)
                logging.debug("Generator: Found sequences: positive: {}, negative: {}".format(len(positive), len(negative)))
                for sequence in positive + negative:
                    sequence_in_data = phase_data.iloc[sequence].index

                    start_in_seconds = data_sample.loc[data_sample.index == sequence_in_data[0]]['seconds'].values
                    end_in_seconds = data_sample.loc[data_sample.index == sequence_in_data[-1]]['seconds'].values
                    self.data_visualizer.plot_vertical_line_at_position(grid_position=(self.injected_sample_iterator, 0),
                                                                    x_position=start_in_seconds,
                                                                    color='green')
                    self.data_visualizer.plot_vertical_line_at_position(grid_position=(self.injected_sample_iterator, 0),
                                                                    x_position=end_in_seconds,
                                                                    color='violet')

    def _get_injector_title(self, selected_anomaly_injector : AbstractManipulator, selected_phases, selected_sensor):
        """
            used this during bug fixing to get the injector title
        """
        title = str(selected_anomaly_injector)
        if selected_anomaly_injector.get_on_selected_phases():
            title += f" | Phases: {selected_phases}"
        if selected_anomaly_injector.get_on_selected_sensors():
            title += f" | Sensor: {selected_sensor}"
        return title
