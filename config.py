import configparser

parser = configparser.ConfigParser()
parser.read('.env.ini')

def config(key: str):
    try:
        return parser.get('env', key)
    except:
        return None
