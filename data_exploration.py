import sys
sys.path.append('..')
import options as op
from data_loader import DataLoader
from data_visualizer import DataVisualizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pprint import pprint
"""
    logging setup
    this is global (I think), so we only have to do it in the main file
"""
import logging
logging.basicConfig(level=logging.INFO,
                        #format="%(asctime)s [%(levelname)s] %(message)s",
                        format = '%(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)




"""
NOTES:
    data manipulation methods is dependent on absolute values

TODO:
    explore anomalies in f1 dataset



columns:
Index(['Seriennummer', 'LOGCHARGEDATETIME', 'batch_id', 'PosixTime', 'GTist',
       'GTsoll', 'Fist', 'B4', 'HLH', 'DHzg', 'B2_ist', 'PCB', 'TDGist',
       'UmwaelzPStrom', 'MoLei1', 'MoLei2', 'DD', 'DDoffs', 'SWVER', 'ECO',
       'phase', 'subphase', 'seconds', 'c_pump_max_60_sec',
       'c_pump_moving_avg'],
most important: 
   GT_ist, B4, B2_ist

"""




def main():
    data_loader = DataLoader()


    # f1_data = data_loader.load_f1_data()
    # f1_data_grouped = data_loader.preprocess_data(f1_data)
    # [
    #     print(f"Group: {group}") 
    #     for group in f1_data_grouped.groups
    # ]
    # return




    large_train_data = data_loader.load_large_train_data()
    large_train_data_grouped = data_loader.preprocess_data(large_train_data)

    logging.info("Data loading and preprocessing done.")

    data_visualizer = DataVisualizer((1,2))
    print(large_train_data_grouped.size())

    
    list_of_groups = list(large_train_data_grouped.groups)
    first_group = large_train_data_grouped.get_group(list_of_groups[0])
    
    x_column = 'seconds'
    y_column = 'GTist'

    # make new column
    first_group['GTist'] = first_group['GTist']*0.1 
    first_group['GTist_anomaly_1'] = first_group['GTist'] + 10
    length = first_group.shape[0]
    
    # define function f(x) = 0.00001x + 10
    f = lambda x: 0.001*(x- first_group['GTist'].min())

    first_group['GTist_anomaly_2'] = f(np.arange(length)) + first_group['GTist']

    sensor_columns = ['GTist', 'GTist_anomaly_1', 'GTist_anomaly_2']
    data_visualizer.plot_at_grid_position(grid_position=(0,0), 
                                          data= first_group,
                                          x_column = x_column,
                                          y_column = sensor_columns,
                                          add_phase_lines=True)

    
    # data_visualizer.plot_at_grid_position(grid_position=(0,1),
    #                                         data= largest_group,
    #                                         x_column = x_column,
    #                                         y_column = y_column,
    #                                         add_phase_lines=True)
    




    
    data_visualizer.show_data()
    




if __name__ == "__main__":
    main()