import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import warnings
import sys
from pathlib import Path

from data_loader import config_loader, data_preprocessing
from logging_util.logger import get_logger
from models import estimators

logger = get_logger(__name__)
config = config_loader.load_config()

def shap_vals(estimator, dataset_x: pd.DataFrame, path = None, it = 1, plot_=True):
    logger.debug(f'Shap values.')
    explainer = shap.KernelExplainer(estimator.model.predict, dataset_x)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        shap_values = explainer.shap_values(dataset_x)
        print('shap values shape: ', np.array(shap_values).shape)

    if plot_:
        fig = plt.figure()
        shap.summary_plot(shap_values, dataset_x, show=False, color_bar=True)
        if path:
            plt.savefig(path / f'shap_values_{it}.png', bbox_inches='tight')
        plt.close()

    return shap_values, dataset_x.index


def apply_shap(model_name: str = 'HistGradientBoostRegressor', fold: int = 0, target: str = 'N_avg_app', 
               filepath: str = '', test: bool = True, subsample_n: int = 0, save: bool = True, 
               resultspath: str = ''):
    
    # set path
    if not filepath:
        filepath = Path(__file__).parent.parent.parent / 'results_corrected' / target

    # load input data
    dtype = data_preprocessing.get_data_types(config)
    x = pd.read_csv(filepath / f'data/{"test" if test else "train"}_set_{fold}.csv', index_col=0, dtype=dtype)
    x = data_preprocessing.one_hot_encoding(x)
    if subsample_n:
        x = x.sample(n=subsample_n, random_state=42)

    # load estimator
    estimator_cls = estimators[model_name]
    estimator = estimator_cls(outpath=filepath)
    estimator = estimator.load(path=estimator.output_path, it=fold)

    # set resultspath
    if not resultspath:
        resultspath = filepath / estimator.abbrev
    else:
        resultspath = Path(resultspath) / target / estimator.abbrev
    resultspath.mkdir(parents=True, exist_ok=True)

    # calculate shap values
    shap_values, ix = shap_vals(estimator, x, filepath, fold, plot_=False)

    # save shap values
    if save:
        shap_values_df = pd.DataFrame(shap_values, index=ix, columns=x.columns)
        shap_values_df.to_csv(resultspath / f'shap_values_{fold}.csv')

    return shap_values, ix, x.columns