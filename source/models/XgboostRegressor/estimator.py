from __future__ import annotations

import pandas as pd
import xgboost as xgb
import joblib
import os
from pydantic import BaseModel, parse_obj_as
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.gaussian_process import *

from data_loader.data_preprocessing import DataSet, one_hot_encoding
from data_loader.config_loader import load_config
from evaluation.evaluation import (
    get_evaluation_metrics_reg,
    plot_prediction,
)
from evaluation.feature_importance import shap_vals
from logging_util.logger import get_logger
from models.base import Estimator

logger = get_logger(__name__)
data_config = load_config(fertilizer=os.getenv('TARGET', ''))


class ParamGrid(BaseModel):
    max_depth: list[int]
    n_estimators: list[int]
    colsample_bytree: list[float]
    subsample: list[float]
    min_child_weight: list[int]
    enable_categorical: list[bool]


class Config(BaseModel):
    param_grid: ParamGrid
    cv: int
    n_jobs: int
    verbose: bool


class XGBoostEstimator(Estimator):
    configcls = Config
    abbrev = 'XGB'

    def preprocess(self, dataset: DataSet) -> DataSet:
        logger.debug(f"Preprocessing data.")

        # XGBoost regressor is trained with CV, so we can combine train and validation set
        X_train = pd.concat([dataset.X_train, dataset.X_valid], axis=0)
        y_train = pd.concat([dataset.y_train, dataset.y_valid], axis=0)
        
        # one hot encoding and standardizing
        if len(data_config.categorical_var):
            X_train = one_hot_encoding(X_train, dataset)
            X_test = one_hot_encoding(dataset.X_test, dataset)
        else:
            X_test = dataset.X_test

        self.feature_names = X_train.columns

        return DataSet(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=dataset.y_test,
            X_valid=None,
            y_valid=None,
        )


    def fit(self, dataset: DataSet) -> XGBoostEstimator:
        logger.debug(f"Training model.")
        hyperparam = self.config.dict()
        if self.cv:
            hyperparam['cv'] = self.cv

        self.grid = GridSearchCV(xgb.XGBRegressor(), 
                                 **hyperparam, 
                                 scoring='neg_mean_absolute_error'
                                 ) 
        self.grid.fit(dataset.X_train, dataset.y_train)
        self.model = self.grid.best_estimator_
        self.inner_mae = -1 * self.grid.best_score_
        self.y_pred = pd.Series(
            self.model.predict(dataset.X_test),
            index=dataset.X_test.index,
            name='predicted_' + data_config.target,
        )
        return self

    def get_feature_importance(self, estimator_name, dataset_x: pd.DataFrame, it: int=0, plot_: bool=True, shap_: bool=False) -> tuple:
        
        shap_vals_, shap_ix = None, None
        if shap_:
            shap_vals_, shap_ix = shap_vals(self, dataset_x, self.output_path, it, plot_)

        return shap_vals_, shap_ix
    

    def get_evaluation_metrics(self, dataset: DataSet, set: str) -> pd.Series:
        if set=='test':
            y = dataset.y_test
            y_pred = self.y_pred
        elif set=='train':
            y = dataset.y_train
            y_pred = pd.Series(self.model.predict(dataset.X_train), index=dataset.X_train.index,
                                     name='predicted_' + data_config.target)
        return get_evaluation_metrics_reg(y, y_pred).rename('xgboost')

    def plot_prediction(self, dataset: DataSet, df: pd.DataFrame, it: int=0) -> None:
        plot_prediction(dataset.y_test, self.y_pred, self.output_path, it)


    def save(self, it: int=0) -> None:
        # pickle grid, model, y_pred and config
        joblib.dump(
            {
                'model': self.model,
                'grid': self.grid,
                'y_pred': self.y_pred,
                'config': self.config.__dict__,
            },
            self.output_path / f'model_{it}.pkl',
        )
        
    def save_predictions(self, it: int=0) -> None:
        self.y_pred.to_csv(self.output_path / f'predictions_{it}.csv')

    @classmethod
    def load(cls, path: str | Path, it: int=0) -> XGBoostEstimator:
        model = joblib.load(path / f'model_{it}.pkl')
        estimator = XGBoostEstimator()
        # overwrite attributes
        estimator.config = parse_obj_as(cls.configcls, model['config'])
        estimator.model = model['model']
        estimator.grid = model['grid']
        estimator.y_pred = model['y_pred']
        return estimator
