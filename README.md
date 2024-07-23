Description
A description of the different folders is given below:

DATA: This folder contains input data for descriptors and data from palladium-catalyzed  cross-coupling reaction reported by Plenio et al.
Model: This folder contains models
Train_test: This folder contains model results

Install Requirements
Please run the following commands to install all requirements:

conda create --name myenv --file environment.yml
This will create a new conda environment with the specified packages with PyTorch. Make sure the PyTorch and DGL versions are compatible with each other and with your CUDA version if you plan to use GPU acceleration.

conda create -n myenv python=3.8
conda activate myenv
conda install pip
pip install -r requirements.txt
This will set up your environment with the specified packages, using PyTorch as the backend for DGL. Make sure the versions specified are compatible with each other to avoid any conflicts.
