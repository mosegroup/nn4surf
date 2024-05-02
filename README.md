# nn4surf
This is a project to approximate driving force terms in surface dynamics simulations using Neural Networks.

# Code dependencies
The code leverages the following python modules:
* numpy (version 1.26.4 tested)
* pytorch (version 2.2.2+cu121 tested)
* matplotlib (version 3.5.1 tested)
* scipy (version 1.12.0 tested)

pip installation should work "out of the box" (consider using a virtual environment)

# Folder structure
The code is organized in the following folders:
* _datasets/_ folder meant to contain training/validation sets
* _in/_ folder containing input files (e.g. initial profiles for evolutions using the trained NN approximation)
* _out/_ folder containing the output of simulations
* _src/_ folder containing source code for classes definition, argument parser for training and utilities
* _trainin_logs/_ folder containing training logs (model snapshots at different epochs, images and train/validation losses)
* _models/_ folder meant to contain trained models

Remember that trained models are torch.nn.Modules objects, and as such they can be loaded/saved/reused as needed!

# Scripts
In the master folder there are the following scripts
* _train.py_ performs training using the specified training and validation sets; a basic argument parser is implemented (run _python3 train.py --help_ for a quick description of the available options)
* _run_evo.py_ runs a simulation using a trained NN model (*NOTE*: at the moment, there is no argument parser: simulation parameters are currently directly modified in the script main function. A future release will add an argument parser also for this script)
* _run_minimizer.py_ performs the minimization of a given initial profile using the NN model (*NOTE*: at the moment, there is no argument parser: simulation parameters are currently directly modified in the script main function. A future release will add an argument parser also for this script)

For additional information, please, contact us (https://github.com/dlanzo).
Happy learning!
