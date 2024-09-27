# Project File Tree
.
├── Report.pdf            # description of the project's design and results
├── client.py             # Client-side code for connecting to the server and receiving global model parameters
├── dataset.py            # Module for dataset handling, used to load training and testing datasets
├── fl_system.py          # Defines the core components of the federated learning system, including the basic framework for training processes
├── model.py              # Module for model definition, containing the definition of the LeNet model
├── param.yaml            # Configuration file containing parameters required during training
└── test_device.py       # Script for testing the availability of MPS, checking if the service is working properly

# Usage
First you should move the `FL_Data` directory to the parent directory of your current location.

Please note that you can specify the device by setting the device field in the param.yaml file to either cuda or mps.
```shell
python test_device.py
```
If the mps device is selected in the param.yaml file, please run the following command to test if mps is available on the current device.

```shell
python fl_system.py --config <the directory of param.yaml>
```


# Parameters
See param.yaml for details.