from __future__ import annotations
import pathlib
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
from pydantic import BaseModel, parse_obj_as
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader.config_loader import load_config, DataConfig
from logging_util.logger import get_logger

config = load_config(fertilizer=os.getenv('TARGET', ''))
logger = get_logger(__name__)


class DataSet(BaseModel):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_valid: pd.DataFrame | None
    y_valid: pd.Series | None

    class Config:
        arbitrary_types_allowed = True

    def save(self, path: Path) -> None:
        joblib.dump(self.__dict__, path / 'dataset.pkl')

    @classmethod
    def load(self, path: Path) -> DataSet:
        return parse_obj_as(DataSet, joblib.load(path / 'dataset.pkl'))


def train_test_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    outpath: pathlib.Path | None = None,
    it: int = 1,
    train_only: bool = False,
    random_state: int = 42,
    train_i = None, 
    test_i = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
) -> DataSet:

    if train_only:
        return DataSet(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            X_valid=None,
            y_valid=None,
        )
    
    # split data
    if train_i is not None: # split according to given indices (train_i, test_i)
        X_train = X.iloc[train_i]
        y_train = y.iloc[train_i]
        X_test = X.iloc[test_i]
        y_test = y.iloc[test_i]
        X_valid = None
        y_valid = None

    else: # random train_test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train, 
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
        )

    # save to csv
    if outpath is not None:
        outpath = outpath / 'data'
        outpath.mkdir(parents=True, exist_ok=True)

        pd.concat([X_train, y_train], axis=1).to_csv(
            outpath / f'training_set_{it}.csv', index=True
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            outpath / f'test_set_{it}.csv', index=True
        )
        if isinstance(X_valid, pd.DataFrame):
            pd.concat([X_valid, y_valid], axis=1).to_csv(
                outpath / f'validation_set_{it}.csv', index=True
        )

    return DataSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_valid=X_valid,
        y_valid=y_valid,
    )


def load_all_data(config=config, dtype=None):

    datapath = Path(__file__).parent.parent.parent.resolve() / 'data'
    if dtype is None:
        dtype = get_data_types(config)

    df = pd.read_csv(datapath / 'FUBC_full_dataset_v3.csv', dtype=dtype, usecols=dtype.keys())

    return df


def one_hot_encoding(data: pd.DataFrame, dataset: DataSet | pd.DataFrame = None) -> pd.DataFrame:
    """
    onehot encode categorical variables
    """

    all_data = load_all_data()

    ohe = OneHotEncoder(categories=[np.unique(all_data[var]) for var in config.categorical_var])
    tmp = ohe.fit_transform(np.array(data[config.categorical_var])).toarray()

    encoded_var = list(ohe.get_feature_names_out(config.categorical_var))  # new variable names
    cat_df = pd.DataFrame(tmp, columns=encoded_var, index=data.index)

    # join encoded variables to original dataframe
    data_ohe = pd.concat([data[config.numerical_var], cat_df], axis=1)

    return data_ohe


def scaler(datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    standardize numerical features
    """
    sc = StandardScaler()
    datasets_rescaled = []
    for df in datasets:
        datasets_rescaled.append(
            pd.DataFrame(sc.fit_transform(df), index=df.index, columns=df.columns)
        )
    return datasets_rescaled


def get_data_types(config: DataConfig) -> dict:

    dtype_cat = {
        key: str for key in config.categorical_var
    }

    dtype_num = {
        key: np.float64 for key in config.numerical_var
    }

    dtype = {**dtype_cat, **dtype_num}
    return dtype