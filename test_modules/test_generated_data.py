import pytest
import os 
import pandas as pd

import options_rational as op
from  dataset_generator import DatasetGenerator

class TestGeneratedData():



    @staticmethod
    def test_generated_dataset():
        
        """
            Test for the generated dataset and statistic json
            includes
                - data can be loaded
                - has right shape and type
                - 

        """
        number_of_samples = 2
        
        dataset_generator = DatasetGenerator(number_of_anomaly_samples = number_of_samples, 
                                            max_number_of_sensors = 1,
                                            save_as_pdf = True,
                                            save_file = True,
                                            filename = "rational")
        
        original_dataset = pd.read_parquet(op.F1_DATA_FILE_LOCATION)
        
        dataset_generator.generate_dataset()

        # Load the generated dataset
        generated_dataset = pd.read_parquet(f"{op.GENERATED_DATA_DIRECTORY}/{dataset_generator.filename}.parquet")

        # check if number of samples is correct
        generated_dataset_grouped = generated_dataset.groupby(op.COLUMNS_TO_GROUP_BY)
        assert generated_dataset_grouped.ngroups == number_of_samples



        # Load the generated statistics


        # teardown: delete generated datafiles
        try:
            os.remove(f"{op.GENERATED_DATA_DIRECTORY}/{dataset_generator.filename}.parquet")
            os.remove(f"{op.GENERATED_DATA_DIRECTORY}/{dataset_generator.filename}_statistics.json")
            os.remove(f"{op.GENERATED_DATA_DIRECTORY}/{dataset_generator.filename}.pdf")
        except Exception as e:
            raise e
            

