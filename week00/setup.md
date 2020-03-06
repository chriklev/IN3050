# Setup of environments and packages needed throughout IN3050/4050
This guide is for setting up your own machine with the software needed to work through the weekly and mandatory exercises in IN3050/IN4050.

## Setup with Anaconda
### Download and install either Anaconda or Miniconda
Download the python 3.7 installer for your operating system.

- [Anaconda](https://www.anaconda.com/distribution/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Install anaconda by following [this](https://docs.anaconda.com/anaconda/install/linux/) guide if you are using linux, or [this](https://docs.anaconda.com/anaconda/install/windows/) guide if you are using windows.

If you are using windows, or just want more info on how to use conda, see [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html). Once you got a working shell with conda installed you can continue the guide.


You should now have access to the conda command. For a quick test execute the conda command without any arguments.
```bash
conda
```

### Create environment
Make sure that you are in the directory with the `conda-env.yml` file that came with this setup guide.
To create an environment with the name "in3050", and all required dependencies for this course, one command is sufficient
```bash
conda env create -f conda-env.yml
```
If you want to change the name of the environment you can edit the name in the  `conda-env.yml` file before executing the command.


**Note:** You can create the environment for this course from scratch if you want. If you do, make sure to use python 3.7.x as all the material was made with this version of python in mind.

### Activate the environment
To activate the environment
```bash
conda activate in3050
```
you can now run code for the course in this environment. If you want to get back to you standard python and packages:
```bash
conda deactivate
```

## Test Jupyter notebook
Open a new terminal and activate the `in3050` environment. Change directory to the `week00/` directory and run the command
```bash
jupyter notebook
```
This should open your browser with a new tab containing the jupyter dashboard with an overview of the `week00/` directory. Open the `test-notebook.ipynb` notebook and follow the steps outlined there.
