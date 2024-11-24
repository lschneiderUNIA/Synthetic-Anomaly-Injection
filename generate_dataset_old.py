import pandas as pd
import random

import logging
logging.basicConfig(level=logging.INFO,
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

# ------------------------------------------------
# Injection Setting
# ------------------------------------------------

NUMBER_OF_ANOMALY_SAMPLES = 6

MAX_NUMBER_OF_SENSORS = 1

#------------------------------------------------
# Initilization and Setup
#------------------------------------------------

data_handler = DataHandler()
sensor_list = opt.MOST_IMPORTANT_SENSOR_COLUMNS

phase_index_list = data_handler.get_phase_indices_list()

# set seed
random.seed(42)


phase_function_anomaly_injector = PhaseFunctionAnomaly(data_handler)
by_gradient_aligner = ByGradientAligner(data_handler)
phase_range_changer = PhaseRangeChanger(data_handler)

# anomaly_injectors = [
#     "phase_function_injector" : phase_function_anomaly_injector,
#     #"phase_range_changer" : phase_range_changer
# ]

anomaly_injector_list = [
    phase_function_anomaly_injector
]

aligner_list = [
    by_gradient_aligner,
    phase_function_anomaly_injector
]

visual_rows = 3
visual_columns = NUMBER_OF_ANOMALY_SAMPLES // visual_rows + 1

data_visualizer = DataVisualizer((visual_rows, NUMBER_OF_ANOMALY_SAMPLES))

logging.INFO("Setup complete")
