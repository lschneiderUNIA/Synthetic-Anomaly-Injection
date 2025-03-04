import argparse

from dataset_generator import DatasetGenerator


def main():

    parser = argparse.ArgumentParser(description="Generate a dataset with synthetic anomalies.")

    parser.add_argument("--number_of_samples", 
                        metavar="The number of samples in the new dataset.", 
                        type=int, 
                       # nargs="+", 
                        help="The number of samples in the new dataset.",
                        default=10)

    args = parser.parse_args()

    number_of_anomaly_samples = args.number_of_samples
    print(number_of_anomaly_samples)
    assert isinstance(number_of_anomaly_samples, int), "The number of samples must be an integer."

    assert number_of_anomaly_samples > 0, "The number of samples must be greater than 0."

    dataset_generator = DatasetGenerator(number_of_anomaly_samples =  args.number_of_samples, 
                                            max_number_of_sensors = 1,
                                            save_as_pdf = True,
                                            save_file = True,
                                            filename = "rational")
    dataset_generator.generate_dataset()

    
if __name__ == "__main__":
    main()
    