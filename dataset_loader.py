import os
import csv
from stl_converter import STLConverter

class DatasetLoader:
    params_file = 'Parameters.csv'
    def __init__(self) -> None:
        self.converter = STLConverter()
        self.y_params = {}

    def load_params(self, directory: str):
        with open(os.path.join(directory, self.params_file), mode='r') as infile:
            reader = csv.reader(infile)
            next(reader) # skip header

            for rows in reader:
                key = rows[0]
                values = rows[1:]
                self.y_params[key] = [float(v) for v in values]

    def load(self, directory: str):
        dataset = []
        self.load_params(directory)
        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if os.path.join(directory, filename) == os.path.join(directory, self.params_file):
                pass
            elif os.path.isfile(os.path.join(directory, filename)):
                stl_file = os.path.join(directory, filename)
                obj_file = self.converter.convert_to_binary(stl_file)
                graph = self.converter.convert_to_graph(obj_file, self.y_params)
                dataset.append(graph)

        return dataset

