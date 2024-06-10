from importlib import import_module

from logging_util.logger import get_logger
from models.base import Estimator

logger = get_logger(__name__)

enabled_estimators = [
        #'LogisticRegressor.estimator.LogiticEstimator',
        #'SVRegressor.estimator.SupportVectorEstimator',
        # 'RandomForestRegressor.estimator.RandomForestEstimator',
        'HistGradientBoostRegressor.estimator.HistGradientBoostEstimator',
        'XgboostRegressor.estimator.XGBoostEstimator',
    ]

estimators: dict[str, Estimator] = {}
for estimator in enabled_estimators:
    estimator_name = estimator.split(".", maxsplit=1)[0]
    logger.info(f'Loading {estimator_name}')
    estimator_module_name, estimator_class = estimator.rsplit(".", maxsplit=1)
    estimator_module = import_module(f".{estimator_module_name}", __name__)
    estimators[estimator_name] = getattr(estimator_module, estimator_class)
