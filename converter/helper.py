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
            head, tail = os.path.split(input_filename)
            output_path = os.path.join(head, output_subdir)

            # create the new file in obj path
            filename = tail.split(".")[0]
            full_filename = f"{filename}.{format}"
            output_filename = os.path.join(output_path, full_filename)

        return output_filename
