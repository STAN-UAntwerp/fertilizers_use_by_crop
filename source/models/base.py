from __future__ import annotations
import sys
from pathlib import Path
from typing import Type, TypeVar, Union
import anyconfig
import pandas as pd
from pydantic import parse_obj_as

from data_loader.data_preprocessing import DataSet

ConfigType = TypeVar('ConfigType')


class Estimator:
    configcls: Type[Union[ConfigType, None]] = type(None)

    @classmethod
    def load_config(cls, configcls: Type[Union[ConfigType, None]]):
        module_path = Path(sys.modules[cls.__module__].__file__)
        raw_config = anyconfig.load(module_path.parent.resolve() / 'config.yaml')
        return parse_obj_as(configcls, raw_config)

    def __init__(
        self, outpath = None, cv: int=None
    ):
        self.config = self.load_config(self.configcls)
        self.abbrev = self.abbrev
        self.cv = cv
        if outpath:
            self.output_path = outpath / self.abbrev
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = None

    def preprocess(self, dataset: DataSet) -> DataSet:
        raise NotImplementedError

    def fit(self, dataset: DataSet) -> Estimator:
        raise NotImplementedError

    def plot_prediction(self, dataset: DataSet, df: pd.DataFrame, it: int=0) -> None:
        raise NotImplementedError

    def get_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_evaluation_metrics(self, dataset: DataSet, set: str) -> pd.Series:
        raise NotImplementedError

    def save(self, it: int=0) -> None: # path: str | Path,
        raise NotImplementedError
    
    def save_predictions(self, it: int=0) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> Estimator:
        raise NotImplementedError
