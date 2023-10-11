class Logger:
    @classmethod
    def _log(self, tag, *str):
        print(f"{tag}: "),
        for i in str:
            print(i),
        print()

    @staticmethod
    def error(*str):
        Logger._log("ERROR", str)

    @staticmethod
    def info(*str):
        Logger._log("INFO", str)
