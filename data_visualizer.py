import matplotlib.axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib




class DataVisualizer():
    """
        Base class for data visualization

        We want to support:
            - singular plots for one time series
            - grid plots for multiple time series
            - optional color coding for phases
            - marking of phases as vertical lines
            - comparing multiple time series in the same plot (ie axes/subplot)
            - visualizing anomalies when comparing two time series

    
        Public Functions:
            - init to define setup: layout, 
            - 

        
    """
    def __init__(self, layout : tuple[int, int]) -> None:
        """
            define format, layout of plots,

        """

        self.layout = layout
        try:
            self.fig, self.axes = plt.subplots(layout[0], layout[1])
        except:
            raise ValueError("Layout must be a tuple of two integers")
        #self.fig.tight_layout()




    def _get_correct_axes(self, grid_position : tuple[int, int]) -> matplotlib.axes:
        """
            get axes object at grid position
            I always want to access the axes object with a tuple, even if its a 1x1 grid or a list of axes objects
        """
        #print(grid_position)
        if grid_position[0] >= self.layout[0] or grid_position[1] >= self.layout[1]:
            raise ValueError("Grid position out of bounds")
        if self.layout[0] == 1 and self.layout[1] == 1:
                return self.axes
        elif self.layout[0] == 1:
            return self.axes[grid_position[1]]
        elif self.layout[1] == 1:
            return self.axes[grid_position[0]]      
        else:
            return self.axes[grid_position[0], grid_position[1]]

    def plot_at_grid_position_with_distinct_points(self, 
                              grid_position : tuple[int, int], 
                              data : pd.DataFrame,  
                              x_column : str, 
                              y_column, # can be str or list[str] for multiple columns
                              distinct_points : list,
                              plot_color :str,
                              x_label : str = None) -> None:
        """
            plot data at grid position, but visualize with distinct points marked in the plot
        """
        axes = self._get_correct_axes(grid_position)
        
        x = data[x_column]
        y = data[y_column]
        axes.set_ylim([0, 200])
        axes.set_xlim([0, 13000])

        if plot_color != None:
            axes.plot(x, y, color=plot_color, label=y_column)
        else:
            axes.plot(x, y)

        if x_label != None:
            axes.set_xlabel(x_label)
        
        axes.set_ylabel(y_column)

        # highlight by a thick dot with plot_color
        for point in distinct_points:
            #axes.scatter(x=point, y=data[y_column].iloc[point], color=plot_color, s=100)
            # make line thinner
            axes.axvline(x = point, color='black', linestyle='--', linewidth=0.5)

                              


    def plot_at_grid_position(self, grid_position : tuple[int, int], 
                              data : pd.DataFrame,  
                              x_column : str, 
                              y_column, # can be str or list[str] for multiple columns
                              x_label : str = None,
                              plot_color :str = None,
                              add_phase_colors : bool = False, 
                              add_phase_lines : bool = False,
                              y_limits : tuple = (0,200),
                              x_limits : tuple = (0,13000))-> None:
        """
            plot data at grid position
            the y columns list is supported by panda and matplotlib and does not need to handled seperately 

        """
        axes = self._get_correct_axes(grid_position)
        axes.set_ylim(y_limits)
        axes.set_xlim(x_limits)
        
        x = data[x_column]
        y = data[y_column]

        if plot_color != None:
            axes.plot(x, y, color=plot_color, label=y_column)
        else:
            axes.plot(x, y)

        if x_label != None:
            axes.set_xlabel(x_label)
        
        axes.set_ylabel(y_column)

        #self._plot_data_at_axes_object(axes, x, y, add_phase_colors)
        # add legend

        #axes.legend(y_column)

        if add_phase_lines:
            self._add_phase_lines(axes, data)



    def _plot_data_at_axes_object(self, axes : matplotlib.axes,  x: str, y: str, add_phase_colors : bool = None) -> None:
        """
            pretty sure its not str objects

            actual plotting function, only gets x,y and axes object
        """
        axes.plot(x, y)


    def _add_phase_lines(self, axes : matplotlib.axes, data : pd.DataFrame, color : str = None) -> None:
        """
            add vertical lines to axes object at phase changes
        """
        if color == None:
            color = 'k'

        phases = data['phase']
        for phase in phases.unique():
            index = np.where(phases == phase)[0][0]
            seconds_index = data['seconds'].iloc[index]
            axes.axvline(x = seconds_index, color=color, linestyle='--')

    def show_data(self) -> None:
        """
            show all plots
        """

        plt.show()