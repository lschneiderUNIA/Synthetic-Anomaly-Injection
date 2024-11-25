import matplotlib.axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import itertools
import math
import logging


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

        self._limit_min_adjustment = 0.8
        self._limit_max_adjustment = 1.2




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
                              y_limits : tuple = (None, None),
                              x_limits : tuple = (None, None),
                              plot_label : str = None,
                              title : str = None)-> None:
        """
            plot data at grid position
            the y columns list is supported by panda and matplotlib and does not need to handled seperately 

        """
        axes = self._get_correct_axes(grid_position)
        
        
        x = data[x_column]
        y = data[y_column]

        self._handle_limits_at_grid_position(x, y, axes, x_limits, y_limits)
        
        line, = axes.plot(x, y)
        # plot horizontal line at y=0
        axes.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        if plot_label != None: 
            line.set_label(plot_label)
            axes.legend()
        if plot_color != None:
            line.set_color(plot_color)


        if x_label != None:
            axes.set_xlabel(x_label)
        
        axes.set_ylabel(y_column)

        #self._plot_data_at_axes_object(axes, x, y, add_phase_colors)
        # add legend

        #axes.legend(y_column)

        if add_phase_lines:
            self._add_phase_lines(axes, data)
        
        if title:
            axes.set_title(title)
    

    def set_title_at_position(self, grid_position : tuple[int, int], title : str) -> None:
        """
            set title at grid position
        """
        axes = self._get_correct_axes(grid_position)
        axes.set_title(title)

    def _handle_limits_at_grid_position(self, 
                                        x, # should be pandas series
                                        y, 
                                        axes : matplotlib.axes,
                                        x_limits : tuple,
                                        y_limits : tuple,) -> None:
        """
            handle limits for axes object
            set new min/max on axes with min and max adjustment to make sure the data is more visible
            this may be a bit overcomplicated, but the visaualization should be as good as possible and doesnt need
            to be efficient

            @param
                x : pandas series
                y : pandas series
                axes : matplotlib.axes
                x_limits : tuple -- can manually set limits what it should be at least
                y_limits : tuple 

        """
        x_min = self._handle_limit_adjustment(min(x), 'min')
        x_max = self._handle_limit_adjustment(max(x), 'max')
        y_min = self._handle_limit_adjustment(min(y), 'min')    
        y_max = self._handle_limit_adjustment(max(y), 'max')

        current_x_limits = axes.get_xlim()
        current_y_limits = axes.get_ylim()

        logging.debug(f"current_x_limits: {current_x_limits}")
        logging.debug(f"current_y_limits: {current_y_limits}")

        # set new limits based on x and y values
        if x_min < current_x_limits[0]:
            axes.set_xlim([self._handle_limit_adjustment(x_min, 'min'), axes.get_xlim()[1]])
        if x_max > current_x_limits[1]:
            axes.set_xlim([axes.get_xlim()[0], self._handle_limit_adjustment(x_max, 'max')])
        if y_min < current_y_limits[0]:
            axes.set_ylim([self._handle_limit_adjustment(y_min, 'min'), axes.get_ylim()[1]])
        if y_max > current_y_limits[1]:
            axes.set_ylim([axes.get_ylim()[0], self._handle_limit_adjustment(y_max, 'max')])
        
        # check if manual limits are set, but only set them if they are outside of the current limits
        if x_limits[0] != None and x_limits[0] < axes.get_xlim()[0]:
            axes.set_xlim([x_limits[0], axes.get_xlim()[1]])
        if x_limits[1] != None and x_limits[1] > axes.get_xlim()[1]:
            axes.set_xlim([axes.get_xlim()[0], x_limits[1]])
        if y_limits[0] != None and y_limits[0] < axes.get_ylim()[0]:
            axes.set_ylim([y_limits[0], axes.get_ylim()[1]])
        if y_limits[1] != None and y_limits[1] > axes.get_ylim()[1]:
            axes.set_ylim([axes.get_ylim()[0], y_limits[1]])    


    def _handle_limit_adjustment(self,input_limit, min_or_max) -> float:
        """
            adjust the limit by a factor to make the data more visible
            we need to handle negative values as well

            @param
                input_limit : float -- the limit we want to adjust
                min_or_max : str -- 'min' or 'max' to determine if we want to adjust the min or max limit
        """
        if input_limit == 0:
            return 0
        if input_limit < 0 and min_or_max == 'min':
            return input_limit * self._limit_max_adjustment
        if input_limit < 0 and min_or_max == 'max':
            return input_limit * self._limit_min_adjustment
        if input_limit > 0 and min_or_max == 'min':
            return input_limit * self._limit_min_adjustment
        else:
            return input_limit * self._limit_max_adjustment



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
        self.fig.tight_layout()
        plt.show()