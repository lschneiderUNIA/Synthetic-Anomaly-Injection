# Synthethic Anomaly Injection for Phase-based Data
This project is built around an anomaly detection tool for industrial machinery in a predictive maintenance context. The main motivation is based on the problem of few existing anamalous data samples to verify anomaly detection methods. We plan to publish a paper detailing the problem, related results and the methodology employed.

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
Call `python3 dataset_generator.py`.

## Data
The data is stored in a `/data` folder using parquet and can be edited in a options.py file. Currently we use a propriatery dataset that is protected by an NDA. In the future, we will add public datasets.