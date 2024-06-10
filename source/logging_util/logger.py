import logging
import logging.config
import pathlib
import anyconfig

def get_logger(name: str):
    config = anyconfig.load(pathlib.Path(__file__).parent.resolve() / 'config.yaml')
    logging.config.dictConfig(config)
    return logging.getLogger(__name__)