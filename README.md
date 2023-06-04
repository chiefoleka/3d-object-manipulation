# 3d-object-manipulation
This repository will hold the code for a 3D manipulation project

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

## Run the project
```bash
python main.py
```

You can view the STL or generated OBJ file using [3dviewer.net](https://3dviewer.net).

Enjoy!