# COMP 512 Project - Dhruv Mongia

## Startup Instructions on Terminal using Sun Lab Machines

### Cloning and setting up the virtual environment
```
git clone https://github.com/dhruv-97/wireless-sensor-deep-learning
conda create --name comp512 tensorflow scikit-learn pandas numpy matplotlib
```

### On Server
```
cd wireless-sensor-deep-learning
conda activate comp512
python3 multi-server.py
```

### On Client
```
cd wireless-sensor-deep-learning
python3 multi-client.py <server-name>
```

For example, if you are running the server on babbage and you want to run the client on euler, you will type the following command on euler
` python3 multi-client.py babbage `

Note that not all machines on the sunlab have conda installed but I am sure that babbage does so please use babbage as the server and for setting up the conda environment.

