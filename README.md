# 3d-object-manipulation
This repository will hold the code for a 3D object manipulation project. The goal of this project is to take 3D objects and describe the features of the object using ML.

## Structure
```bash
├── consts # model constants to make selecting models to train easy
├── converter
│   ├── binary_to_obj.py # convert binary STL files to OBJ files
│   ├── helper.py
│   ├── stl_to_graph.py # convert the STL file to a torch3d compatible graph object
├── data # where to place your data and then set the right location in .env.ini
├── loader # STL loader that also appends the contents of Parameter.csv to the graph as the expected output
├── models
│   │   ├── factories # A factory class to create and return an instance of a model based on config params
│   ├── basic_norm.py # Models designed to use a combination of x and norm values during training and evaluation
│   ├── basic.py # Models only using x values during training and evaluation
├── training # entry point to train and evaluate the model
├── converter
│   ├── config.py # loads values of .env.ini and makes them available everywhere
│   ├── logger.py
│   ├── plotter.py # plots the output of training and evaluation
├── .env.ini
└── setup.py
```

## Setup with Anaconda
If you have Anaconda installed, [create and activate a virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```bash
conda create --name 3d-object
conda activate 3d-object
```

## Setup without Anaconda
If you are not using Anaconda, [create and activate a Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
```bash
python -m venv venv
source venv/bin/activate
```

Install all the needed packages
```bash
pip install -r requirements.txt
```
> 🛑 As at the time of writing this project, pytorch3d fails installation using pip. My guess is that the failing build resulted in this issue. I had to build pytorch3d from source using the `v0.7.3` version.

Build `pytorch3d` from source:
```bash
pip install pytorch3d @ "git+https://github.com/facebookresearch/pytorch3d.git@35badc0892275c35818ca39800ec55d9c7342c8f"
```

Copy the env file sample and configure it as needed
```bash
cp .env.sample.ini .env.ini
```

## Run the project
```bash
python setup.py install
python training/main.py
```

You can view the STL or generated OBJ file using [3dviewer.net](https://3dviewer.net).

Enjoy!