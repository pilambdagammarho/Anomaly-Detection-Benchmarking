# Supervised vs Unsupervised Anomaly Benchmarking

The aim of this project is to examine the extent of supervision required for supervised and semi supervised models to reach or outperform unsupervised anomaly detection in time series data.\
To that direction, we have implemented a supervised and extended a semi supervised approach to benchmark the performance on [SMAP](https://podaac.jpl.nasa.gov/SMAP?tab=related-links&sections=about%2Bdata) & [MSL](https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MSL-M-REMS-2-EDR-V1.0) Dataset.

Please find the semi supervised extension here: [DeepSAD](https://www.uni-hildesheim.de/gitlab/haiders/anomalybenchmarking)

## Installation

Use the following command to setup the virtual environment and get started.

```bash
pip install -r requirements.txt
```

## Usage

```python
python main.py
```

## Configurations

The project reads and access the configuration file under `configs/config.yaml` to run training and inference. The project supports both `cuda` and `cpu` train and inference mode at the moment.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
