import configparser
from .stdlib.runtime_resources import Resources

def load_config(resources):
    config = configparser.ConfigParser()
    config.read(resources.getpath('config.ini'))
    return config