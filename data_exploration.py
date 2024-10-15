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
   GTist, B4, B2_ist

"""

class DataExplorer():

    def __init__(self) -> None:
        data_loader = DataLoader()

        large_train_data = data_loader.load_large_train_data()
        self.large_train_data_grouped = data_loader.groupby_date_serial_number(large_train_data)

        self.f1_data = data_loader.load_f1_data()
        self.f1_data_grouped = data_loader.groupby_date_serial_number(self.f1_data)
        
        self.f1_labels_file = data_loader.load_f1_labels(self.f1_data)

        self.f1_labels_dict = {(entry['Seriennummer'] , entry['LOGCHARGEDATETIME']): entry for entry in self.f1_labels_file}

    
    def main(self):
        #self.plotting_f1()
        self.average_phases()


    def plotting_f1(self, f1_data : pd.DataFrame) -> None:
        """
            explore the f1 data
        """
   
        plot_grid = (2,8)

        data_visualizer = DataVisualizer(plot_grid)

        anomaly_color = 'red'
        normal_color = 'blue'

        anomaly_counter = 0
        normal_counter = 0
        
        for i,group in enumerate(self.f1_data_grouped.groups):
            # if i <15:
            #     continue


            data= self.f1_data_grouped.get_group(group)
            
            serial_number = data.iloc[0]['Seriennummer']
            date = data.iloc[0]['LOGCHARGEDATETIME'].strftime('%Y-%m-%d %H:%M:%S')

            if self.f1_labels_dict[serial_number, date]['anomaly']  == True:
                pos = anomaly_counter
                grid = (1, pos)
                anomaly_counter += 1
                plot_color = anomaly_color
                
            else:
                pos = normal_counter
                grid = (0, pos)
                normal_counter += 1
                plot_color = normal_color

            if pos >= plot_grid[1]:
                continue
            
            data_visualizer.plot_at_grid_position(grid_position=grid,
                                                    data= data,
                                                    x_column = 'seconds',
                                                    y_column = 'B2_ist',
                                                    x_label = f"{serial_number}-{date}",
                                                    plot_color = plot_color,
                                                    add_phase_lines=True)
            
        print(f"Anomalies: {anomaly_counter}")
        print(f"Normal: {normal_counter}")
        data_visualizer.show_data()


    def average_phases(self):
        """
            explore average of the phases
        """
        list_of_groups = list(self.large_train_data_grouped.groups)
        sample_group = self.large_train_data_grouped.get_group(list_of_groups[0])

        number_of_phases = sample_group['phase'].nunique()
        print(f"Number of phases: {number_of_phases}")

        phase_lengths = 
        



def main():
   
        
    explorer = DataExplorer()
    explorer.main()

    return






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