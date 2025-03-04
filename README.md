# Synthethic Anomaly Injection for Phase-based Data
This project is built around an anomaly detection tool for industrial machinery in a predictive maintenance context. The main motivation is based on the problem of few existing anamalous data samples to verify anomaly detection methods. We plan to publish a paper detailing the problem, related results and the methodology employed.

The main development is done on a private gitlab instance. The progress is only periodically updated here.
## Requirements
Python >= 3.10 <br>
venv package

## Installation

After cloning create environemnt with 
```bash
python3 -m venv .venv
```

Activate environment with 
```bash
source .venv/bin/activate
```

Install the required packages with 
```bash
pip install -r requirements.txt
```

## Usage
Usage currently requires a propriatery dataset bound by an NDA.

Call `python3 generate_dataset_script.py`. Add `--number_of_samples` to specify the number of samples to generate. Default is 10.
This will create 3 files in the `generated datasets`folder:
* parquet file as the dataset
* pdf visualizing the anomalies
* json file which includes 
    * the original sample id
    * information on which phase, sensor and injection method was used
    * metrics

## Data
The data is stored in a `/data` folder using parquet and can be edited in a options.py file. Currently we use a propriatery dataset that is protected by an NDA. In the future, we will add public datasets.