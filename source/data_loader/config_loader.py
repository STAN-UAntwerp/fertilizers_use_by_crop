import anyconfig
import pathlib
from pydantic import BaseModel, parse_obj_as
from logging_util.logger import get_logger

logger = get_logger(__name__)


class DataConfig(BaseModel):
    target: str
    categorical_var: list[str]
    numerical_var: list[str]
    drop_var: list[str]
    csv_files: dict[str, str]
    new_targets: dict[str, str]
    new_features: dict[str, str]
    crop_codes: dict[str, str]

def load_config(fertilizer: str = '') -> DataConfig:
    logger.debug("Loading data configuration")
    config_filename = 'config.yaml' if not fertilizer else f'config_{fertilizer}.yaml'
    raw_config = anyconfig.load(pathlib.Path(__file__).parent.resolve() / config_filename)
    return parse_obj_as(DataConfig, raw_config)
