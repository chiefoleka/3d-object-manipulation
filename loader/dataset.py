import os
import csv

from converter.binary_to_obj import BinaryToObj
from converter.stl_to_graph import StlToGraph
from utils.config import config as get_config


class Dataset:
    params_file = "Parameters.csv"

    def __init__(self) -> None:
        self.binary_converter = BinaryToObj()
        self.obj_converter = StlToGraph()
        self.y_params = {}
        self.y_params_normalized = {}

        self.min_y = { "x": float("inf"), "y": float("inf"), "z": float("inf") }
        self.max_y = { "x": 0, "y": 0, "z": 0 }

    def load_params(self, directory: str):
        with open(os.path.join(directory, self.params_file), mode="r") as infile:
            reader = csv.reader(infile)
            next(reader)  # skip header

            for rows in reader:
                key = rows[0]
                values = rows[1:]
                self.y_params[key] = [float(v) for v in values]
                self.calculate_min_max_y(self.y_params[key])

            self.normalize_y()

    def calculate_min_max_y(self, values):
        self.min_y["x"] = min(self.min_y["x"], values[0])
        self.min_y["y"] = min(self.min_y["y"], values[1])
        self.min_y["z"] = min(self.min_y["z"], values[2])

        self.max_y["x"] = max(self.max_y["x"], values[0])
        self.max_y["y"] = max(self.max_y["y"], values[1])
        self.max_y["z"] = max(self.max_y["z"], values[2])

    def normalize_y(self):
        for key in self.y_params:
            y = self.y_params[key]
            y[0] = (y[0] - self.min_y["x"]) / (self.max_y["x"] - self.min_y["x"])
            y[1] = (y[1] - self.min_y["y"]) / (self.max_y["y"] - self.min_y["y"])
            y[2] = (y[2] - self.min_y["z"]) / (self.max_y["z"] - self.min_y["z"])

            self.y_params_normalized[key] = y

    def load(self, directory: str):
        dataset_list = []
        self.load_params(directory)

        # ensure the obj directory exists so we can save the files
        output_subdir = get_config("output_subdir")
        full_path = os.path.join(directory, output_subdir)
        os.makedirs(full_path, exist_ok=True)

        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if os.path.join(directory, filename) == os.path.join(
                directory, self.params_file
            ):
                pass
            elif os.path.isfile(os.path.join(directory, filename)):
                stl_file = os.path.join(directory, filename)
                obj_file = self.binary_converter.convert_to_obj(stl_file)
                graph = self.obj_converter.convert_to_graph(obj_file, self.y_params)
                dataset_list.append(graph)

        return dataset_list
