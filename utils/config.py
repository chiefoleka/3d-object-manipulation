import configparser

parser = configparser.ConfigParser()
parser.read('.env.ini')

def config(key: str, default = None):
    try:
        return parser.get('env', key)
    except:  # noqa: E722
        return default
