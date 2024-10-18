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
from scipy.interpolate import interp1d


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
        #self.average_phases_with_interpolation(selected_sensor= 'B2_ist')

        # 2x4 grid, with rows being sensors, and columns being different anomaly factors and phase_shift_lengths
        data_visualizer = DataVisualizer((2,4))

        sensor_list = ['GTist', 'B4']


        anomaly_factor_list = [0.4, 0.7, 1.2, 1.5]
        phase_shift_length_factor_list = [0.9] * 4
        for row, sensor in enumerate(sensor_list):
            for i, (anomaly_factor, phase_shift_length_factor) in enumerate(zip(anomaly_factor_list, phase_shift_length_factor_list)):
                self.phase_test_anomaly(sensor=sensor,
                                        grid_position=(row,i),
                                        data_visualizer=data_visualizer,
                                        anomaly_factor=anomaly_factor,
                                        phase_shift_length_factor=phase_shift_length_factor)


        data_visualizer.show_data()

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


    def average_phases_with_interpolation(self, selected_sensor):
        """
            explore average of the phases
        """
        list_of_groups = list(self.large_train_data_grouped.groups)
        sample_group = self.large_train_data_grouped.get_group(list_of_groups[0])

        # phase dict to store all phase lengths
        phase_indices = sample_group['phase'].unique()
        phase_dict_lengths = {phase: [] for phase in phase_indices}
        phase_dict_dataframes = {phase: [] for phase in phase_indices}
        
        # iterate over all groups
        # add phase lengths to phase_dict
        # add dataframes to phase_dict
        for i, group in enumerate(list_of_groups):
            data = self.large_train_data_grouped.get_group(group)
            # get phase in data for each phase_index
            # store phase length in appropriate dict entry
            for phase_index in phase_indices:
                phase = data.loc[data.phase == phase_index]
                phase_length = phase['seconds'].max() - phase['seconds'].min()

                phase_dict_lengths[phase_index].append(phase_length)
                phase_dict_dataframes[phase_index].append(phase)

        entries_per_phase = [len(phase_dict_lengths[phase]) for phase in phase_dict_lengths]
        #check that all dict entries have same length
        # just checks that all time series have the same phases
        assert all([entry == entries_per_phase[0] for entry in entries_per_phase]) == True
        print(entries_per_phase)

        average_phase_lengths = {phase: int(np.mean(phase_dict_lengths[phase])) for phase in phase_dict_lengths}
        print(average_phase_lengths)
        phase_interpolation_spaces = {
            phase: np.linspace(0, length, length) for phase,length in average_phase_lengths.items()
        }   

        phase_dict_interpolated = {phase: [] for phase in phase_indices}

        for phase, dataframes in phase_dict_dataframes.items():
            for data in dataframes:
                x = data['seconds'].max() - data['seconds']
                y = data[selected_sensor]
                interpolator = interp1d(x, y, kind='linear', fill_value="extrapolate")
                interpolated_values = interpolator(phase_interpolation_spaces[phase])
                phase_dict_interpolated[phase].append(interpolated_values)
                #interpolated_values = np.interp(common_time_points, normalized_time, phase['value'])

        average_phases = {phase: np.flip(np.mean(phase_dict_interpolated[phase], axis=0)) for phase in phase_indices}
        #pandas dataframe from average series
        # with phase column
        # and seconds column

        average_series_df = pd.DataFrame()
        for i, phase in enumerate(phase_indices):
            current_length = len(average_series_df.index)
            assert isinstance(current_length, int)
            average_series_df = pd.concat([average_series_df, pd.DataFrame({
                'phase': [phase]*len(average_phases[phase]),
                'seconds': range(current_length, current_length+average_phase_lengths[phase]),
                selected_sensor: average_phases[phase]
            })])
        data_visualizer = DataVisualizer((1,1))
        data_visualizer.plot_at_grid_position(grid_position=(0,0),
                                              data= average_series_df,
                                              x_column = 'seconds',
                                              y_column = selected_sensor,
                                              add_phase_lines=True)
        
        data_visualizer.show_data()

    def phase_test_anomaly(self, sensor,
                             grid_position = (0,0),
                             data_visualizer = DataVisualizer((1,1)),
                             phase = 2,
                             anomaly_factor = 0.3,
                             phase_shift_length_factor = 0.5):
        """
            explore phase anomalies
        """

        def simple_multiply(data, phase, sensor, anomaly_factor):
            data.loc[data.phase == phase, sensor] *= anomaly_factor
            return data
        
        def e_function_multiply(data, phase, sensor, anomaly_factor):
            """
                anomaly_factor is the end multiplier, we want a exponential increase up to this factor
            """
            phase_length = data.loc[data.phase == phase].shape[0]
            function_multiplier = np.log(anomaly_factor) / phase_length
            e_function = lambda x : np.exp(function_multiplier * x)

            data.loc[data.phase == phase, sensor] *= e_function(np.arange(phase_length))

            phase_end_data_point = data.loc[data.phase == phase].iloc[-1]
            print(phase_end_data_point)


            return data

        next_phase = phase + 1


        list_of_groups = list(self.large_train_data_grouped.groups)
        group_index = 0
        # group_index = random.randint(0, len(list_of_groups))  
        sample_group = self.large_train_data_grouped.get_group(list_of_groups[group_index])

        data_visualizer.plot_at_grid_position(grid_position=grid_position,
                                              data= sample_group,
                                              x_column = 'seconds',
                                              y_column = sensor,
                                              add_phase_lines=True,
                                              plot_color='blue')
        


        sample_group = e_function_multiply(sample_group, phase, sensor, anomaly_factor)

        phase_end_data_point = sample_group.loc[sample_group.phase == phase].iloc[-1]
        #print(phase_end_data_point)

        next_phase_start_data_point = sample_group.loc[sample_group.phase == next_phase].iloc[0]
        #print(next_phase_start_data_point)



        next_phase_length = sample_group.loc[sample_group.phase == next_phase].shape[0]


        phase_shift_length = int(next_phase_length * phase_shift_length_factor)

        initial_phase_shift = phase_end_data_point[sensor] / next_phase_start_data_point[sensor]

        data_shift = np.logspace(np.log10(initial_phase_shift), np.log10(1.0), phase_shift_length)

        next_phase_sensor_data = sample_group.loc[sample_group.phase == next_phase, sensor][:phase_shift_length]
        print(next_phase_sensor_data)
        next_phase_sensor_data = next_phase_sensor_data * data_shift
        print(next_phase_sensor_data)

        next_phase_rows = sample_group.loc[sample_group.phase == next_phase]

        # Update the sensor column for the first phase_shift_length rows
        sample_group.loc[next_phase_rows.index[:phase_shift_length], sensor] = next_phase_sensor_data        
        print(sample_group.loc[sample_group.phase == next_phase, sensor][:phase_shift_length])



        data_visualizer.plot_at_grid_position(grid_position=grid_position,
                                              data= sample_group,
                                              x_column = 'seconds',
                                              y_column = sensor,
                                              add_phase_lines=True,
                                              plot_color='red')
        #data_visualizer.show_data()


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