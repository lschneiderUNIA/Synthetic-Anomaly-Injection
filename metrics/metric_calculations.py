import numpy as np
import pandas as pd
import dtw
from fastdtw import fastdtw
import matplotlib.pyplot as plt 


class MetricCalculations():
    """
        class for metric calculations
        currently only static methods, dont see a reason for an instance

    
    """

    @staticmethod
    def compute_metric_for_dataframes(
                                    dataframe1 : pd.DataFrame, 
                                    dataframe2 : pd.DataFrame, 
                                    columns : list,
                                    metric_function) -> dict:
        """
            compute metric given by a metric function for each column in the dataframes
        """
        metric_for_each_column = {}
        for column in columns:
            metric_for_each_column[column] = metric_function(dataframe1[column], dataframe2[column])
        #plt.show()

        return metric_for_each_column
        

    @staticmethod  
    def calc_euclidean(first_series, second_series):
        """
            should be useless in our case as dataset magnitues differ to much
        """
        return np.sqrt(np.sum((first_series - second_series) ** 2))

    @staticmethod
    def calc_mape(first_series, second_series):
        """
            mean absolute percentage error
        """
        first_series = MetricCalculations.replace_null_with_zero(first_series)
        second_series = MetricCalculations.replace_null_with_zero(second_series)
        
        return np.mean(np.abs((first_series - second_series) / first_series))
    
    @staticmethod
    def calculate_dtw(first_series, second_series):
        """
            dynamic time warping
        """
        alignment = dtw.dtw(first_series, second_series, keep_internals=True)
        #plot alignment
        #alignment.plot(type="threeway")       
        
        return alignment.distance

    @staticmethod
    def calculate_fast_dtw(first_series, second_series):
        """
            fast dynamic time warping
        """
        distance, path = fastdtw(first_series, second_series)
        return distance
    @staticmethod
    def clean_metric_dictionary(metric_dictionary : dict) -> dict:
        """
            remove 0s and round
        """
        return {sensor: round(metric, 4) for sensor, metric in metric_dictionary.items() if metric != 0}
    

    @staticmethod
    def replace_null_with_zero(dataframe : pd.DataFrame) -> pd.DataFrame:
        """
            replace all NaN values with 0
        """
        return dataframe.fillna(0)