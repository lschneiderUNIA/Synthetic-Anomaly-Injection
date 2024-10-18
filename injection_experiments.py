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



#------------------------------------------------
# Initilization 
#------------------------------------------------

data_handler = DataHandler()
sensor_list = opt.MOST_IMPORTANT_SENSOR_COLUMNS

phase_anomaly_injector = PhaseAnomalyInjector()

data_visualizer = DataVisualizer((1,1))


#------------------------------------------------




group_index = random.randint(0, data_handler.get_number_of_groups())
logging.info("Selected group index: {}".format(group_index))

phase_index = 2
anomaly_factor = 0.5
selected_sensor = sensor_list[0]
logging.info("Selected sensor: {}".format(selected_sensor))

group = data_handler.get_group_by_index(group_index)

# plot original data
data_visualizer.plot_at_grid_position(grid_position=(0,0),
                                        data=group,
                                        x_column='seconds',
                                        y_column=selected_sensor,
                                        add_phase_lines=True,
                                        plot_color='blue')

# inject anomaly
group = PhaseAnomalyInjector.simple_multiply(group,
                                     selected_sensor,
                                     phase_index,
                                     anomaly_factor)



# plot data with anomaly
data_visualizer.plot_at_grid_position(grid_position=(0,0),
                                        data=group,
                                        x_column='seconds',
                                        y_column=selected_sensor,
                                        add_phase_lines=True,
                                        plot_color='red')



new_group = group.reset_index(drop=True)
print(type(new_group))
new_dataframe = pd.DataFrame(group)
print(type(new_dataframe))

group = data_handler.get_group_by_index(group_index)
data_visualizer.plot_at_grid_position(grid_position=(0,1),
                                        data=group,
                                        x_column='seconds',
                                        y_column=selected_sensor,
                                        add_phase_lines=True,
                                        plot_color='green')

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