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
from data_visualizer import DataVisualizer

from data_manipulation.phase_anomaly_injector import PhaseAnomalyInjector
from data_manipulation.phase_alignment import PhaseAligner
from data_manipulation.phase_function_anomaly import PhaseFunctionAnomaly


#------------------------------------------------
# Initilization 
#------------------------------------------------

data_handler = DataHandler()
sensor_list = opt.MOST_IMPORTANT_SENSOR_COLUMNS

phase_anomaly_injector = PhaseAnomalyInjector(data_handler)
phase_aligner = PhaseAligner(data_handler)
phase_function_injector = PhaseFunctionAnomaly(data_handler)

data_visualizer = DataVisualizer((1,1))


#------------------------------------------------
# Setting up
#------------------------------------------------

# get random group
group_index = random.randint(0, data_handler.get_number_of_groups())
group = data_handler.get_group_by_index(group_index)

logging.info("Selected group index: {}".format(group_index))


# get phase indices
phase_index_list = data_handler.get_phase_indices_list()
phase_index_list = phase_index_list[1:2]
logging.info("Selected phase indices: {}".format(phase_index_list))
# select sensor
selected_sensor = sensor_list[0]
logging.info("Selected sensor: {}".format(selected_sensor))

# plot initial data
data_visualizer.plot_at_grid_position(grid_position=(0,0),
                                        data=group,
                                        x_column='seconds',
                                        y_column=selected_sensor,
                                        add_phase_lines=True,
                                        plot_color='blue')

#------------------------------------------------
# ACTUAL EXPERIMENT
#------------------------------------------------

function_parameters = {'type' : 'constant', 'factor' : 0.7}

group = phase_function_injector.inject_function_on_data(
                                function_parameters,
                                group,
                                selected_sensor,
                                phase_index_list)

function_parameters = {'type' : 'linear', 'start_factor' : 1, 'end_factor' : 1.6}

group = phase_function_injector.inject_function_on_data(
                                function_parameters,
                                group,
                                selected_sensor,
                                phase_index_list)

# aligning
function_parameters = {'type' : 'linear'}
group = phase_function_injector.inject_function_on_data(   
                                        function_parameters,
                                        group,
                                        selected_sensor,
                                        [phase_index_list[0]-1],
                                        alignment_factor=0.5,
                                        align_to_next=True)


group = phase_function_injector.inject_function_on_data(
                                        function_parameters,
                                        group,
                                        selected_sensor,
                                        [phase_index_list[-1]+1],
                                        alignment_factor=0.7,
                                        align_to_next=False)



# inject anomaly
# group = phase_anomaly_injector.linear_function(group,
#                                      selected_sensor,
#                                      phase_index,
#                                      anomaly_factor)s
# group = phase_anomaly_injector.linear_function(group,
#                                         selected_sensor,
#                                         phase_index,
#                                         anomaly_factor,)

# group = phase_aligner.phase_alignment('log',   
#                                         group,
#                                         selected_sensor,
#                                         phase_index,
#                                         0.8)


#------------------------------------------------
# FINAL PLOTTING
#------------------------------------------------


# plot data with anomaly
data_visualizer.plot_at_grid_position(grid_position=(0,0),
                                        data=group,
                                        x_column='seconds',
                                        y_column=selected_sensor,
                                        add_phase_lines=True,
                                        plot_color='red')


data_visualizer.show_data()



"""
TODO: test new dataset creation
turn group into a dataframe by resetting indices
new_group = group.reset_index(drop=True)
print(type(new_group))

not sure if this works the same way
new_dataframe = pd.DataFrame(group)
print(type(new_dataframe))

"""