import os


class Helper:
    @staticmethod
    def check_file_is_stl(file):
        if Helper.check_file_exists(file) and file.split(".")[-1].lower() != "stl":
            return False

        return True

    @staticmethod
    def check_file_exists(file):
        if not os.path.exists(file):
            return False

        return True

    @staticmethod
    def ensure_file_exists(
        input_filename, output_filename=None, output_subdir="obj", format="obj"
    ):
        if output_filename is None:
            input_paths = input_filename.split("/")
            output_paths = input_paths[:-1]
            output_paths.append(output_subdir)

            # create the new file in obj path
            filename = input_paths[-1].split(".")[0]
            output_paths.append(f"{filename}.{format}")
            output_filename = "/".join(output_paths)

        return output_filename
