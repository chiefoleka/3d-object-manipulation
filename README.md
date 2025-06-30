# Graph Convolutional Networks for Predicting Mechanical Characteristics of 3D Lattice Structures

This repository will hold the code for a 3D object manipulation project. The goal of this project is to take 3D objects and describe the features of the object using ML.

For All use of the data, please `cite` the following:

_Oleka, Valentine, Seyyed Mohsen Zahedi, Aboozar Taherkhani, Reza Baserinia, S. Abolfazl Zahedi, and Shengxiang Yang. "Graph Convolutional Networks for Predicting Mechanical Characteristics of 3D Lattice Structures." In International Conference on Intelligent Information Processing, pp. 150-160. Cham: Springer Nature Switzerland, 2024_.

The paper is available at: [Graph Convolutional Networks for Predicting Mechanical Characteristics of 3D Lattice Structures | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-031-57919-6_11#citeas).

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
├── utils
│   ├── config.py # loads values of .env.ini and makes them available everywhere
│   ├── logger.py
│   ├── plotter.py # plots the output of training and evaluation
├── .env.ini
└── setup.py
```

## Setup with Anaconda
If you have Anaconda installed, [create and activate a virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```bash
conda create --name 3d-object pip
conda activate 3d-object
```

## Setup without Anaconda
If you are not using Anaconda, [create and activate a Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
```bash
python -m venv venv
source venv/bin/activate
```

On Windows
```bash
.\venv\Scripts\activate
```

Install all the needed packages
```bash
pip install -r requirements.txt
```
> 🛑 As at the time of writing this project, pytorch3d and torch-scatter fails installation using the requirements file. This has to do with changes to how they are packaged and shipped.

Install `torch-scatter` (as it fails to find torch when bundled together - dep management is weird it seems):
```bash
pip install torch-scatter
```

Install `pytorch3d`:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Copy the env file sample and configure it as needed
```bash
cp .env.sample.ini .env.ini
```

## Run the project
```bash
python setup.py install
```

On Unix systems:
```bash
python training/main.py
```

On Windows systems:
```bash
python training\main.py
```

You can view the STL or generated OBJ file using [3dviewer.net](https://3dviewer.net).

> **NOTE: 💡**
>
> If you make any change to the code, always run `python setup.py install` to rebuild the project before running it again.

Enjoy!