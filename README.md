# Project Overview
This project implements a system that simulates training a classification model on the **BloodMNIST** dataset using horizontal **federated learning**.

The dataset is pre-partitioned into a fixed number of clients, each simulating a distinct participant in the federated learning process.

# Project File Tree
```bash
.
├── Report.pdf            # description of the project's design and results
├── client.py             # Client-side code for connecting to the server and receiving global model parameters
├── dataset.py            # Module for dataset handling, used to load training and testing datasets
├── fl_system.py          # Defines the core components of the federated learning system, including the basic framework for training processes
├── model.py              # Module for model definition, containing the definition of the LeNet model
├── param.yaml            # Configuration file containing parameters required during training
└── test_device.py       # Script for testing the availability of MPS, checking if the service is working properly
```

# PipeLine of the system 
<img width="948" alt="image" src="https://github.com/user-attachments/assets/f23bba70-96f5-4f1d-86f6-37814d038efb">


# Usage
Before running the system, move the `FL_Data` directory to the parent directory of your current location.

You can specify the device for computation (either cuda or mps) by setting the appropriate field in the param.yaml configuration file.

To test the device availability, use the following command:

```shell
python test_device.py
```
If the mps device is selected in the param.yaml file, please run the following command to run the whole system.
```shell
python fl_system.py --config <the directory of param.yaml>
```

# Parameters
Details regarding the system’s parameters can be found in the [param.yaml](./param.yaml) configuration file.
