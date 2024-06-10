import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader.data_preprocessing import DataSet
from logging_util.logger import get_logger

logger = get_logger(__name__)


def get_evaluation_metrics_reg(true_values: pd.Series | np.ndarray, predicted_values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(
        {
            'R2': r2_score(true_values, predicted_values),
            'RMSE': mean_squared_error(true_values, predicted_values, squared=False),
            'MSE': mean_squared_error(true_values, predicted_values),
            'MAE': mean_absolute_error(true_values, predicted_values),
        }
    )


class Eval_metrics:
    def __init__(self, num_iterations, estimators, outpath):
        self.num_it = num_iterations
        self.estimator_names = list(estimator.abbrev for estimator in estimators.values()) + ['naive']
        self.outpath = outpath
        self.metric_list = ['MAE', 'RMSE', 'MSE', 'R2']
        self.metrics_test = pd.DataFrame(
            columns = [i for i in range(self.num_it)],
            index = [np.array([metric for metric in self.metric_list for _ in range(len(self.estimator_names))]),
                     np.array((len(self.metric_list)) * self.estimator_names)], 
        )
        self.metrics_train = pd.DataFrame(
            columns = [i for i in range(self.num_it)],
            index = [np.array([metric for metric in self.metric_list for _ in range(len(self.estimator_names))]),
                     np.array((len(self.metric_list)) * self.estimator_names)], 
        )
        self.AE = {}
        for estimator_name in self.estimator_names:
            self.AE[estimator_name] = []
        
        self.labels = {'R2': 'individual R2',
                       'MAE': 'mean absolute error',
                       'MAE': 'mean absolute scaled error',
                       'MSE': 'mean squared error',
                       'SE': 'standard error',
                       'accuracy_score': 'accuracy score'
                      }

        
    def init_plot(self) -> None:
        N = 1 
        self.fig, self.axs = plt.subplots(1, N,
                                          figsize = (5 * self.num_it, 3 * N)) 
        for m, metric in enumerate([self.metric_list[0]]): 
            ax = self.axs if N==1 else self.axs[m]
            ax.set_xticks(np.arange(len(self.estimator_names)), self.estimator_names)
            ax.set_ylabel(self.labels[metric])


    def naive_metrics(self, dataset: DataSet, it: int) -> None:
    
        if 'MAE' in self.metric_list:
            average_test = np.full(len(dataset.y_test), dataset.y_train.mean())
            average_train = np.full(len(dataset.y_train), dataset.y_train.mean())
            for metric in self.metric_list:
                self.metrics_test.loc[(f'{metric}', 'naive'), it] = get_evaluation_metrics_reg(dataset.y_test, average_test)[metric]
                self.metrics_train.loc[(f'{metric}', 'naive'), it] = get_evaluation_metrics_reg(dataset.y_train, average_train)[metric]


    def calculate_metrics(self, estimator, estimator_dataset: DataSet, it: int, plot_: bool=False, verbose: bool=True) -> None:

        self.AE[estimator.abbrev] += np.abs(estimator.y_pred - estimator_dataset.y_test).to_list()

        metric_vals = estimator.get_evaluation_metrics(estimator_dataset, set='test')
        for metric in self.metric_list:
            if metric in metric_vals.index:
                self.metrics_test.loc[(f'{metric}', estimator.abbrev), it] = metric_vals[metric]

        metric_vals = estimator.get_evaluation_metrics(estimator_dataset, set='train')
        for metric in self.metric_list:
            if metric in metric_vals.index:
                self.metrics_train.loc[(f'{metric}', estimator.abbrev), it] = metric_vals[metric]

        if self.num_it > 1:
            self.metrics_test['mean'] = self.metrics_test[[i for i in range(self.num_it)]].mean(axis=1)
            self.metrics_test['std'] = self.metrics_test[[i for i in range(self.num_it)]].std(axis=1)
            self.metrics_test = self.metrics_test[['mean', 'std'] + [i for i in range(self.num_it)]]

            self.metrics_train['mean'] = self.metrics_train[[i for i in range(self.num_it)]].mean(axis=1)
            self.metrics_train['std'] = self.metrics_train[[i for i in range(self.num_it)]].std(axis=1)
            self.metrics_train = self.metrics_train[['mean', 'std'] + [i for i in range(self.num_it)]]
    
        if verbose:
            print(self.metrics_test)
        if self.outpath:
            self.metrics_test.to_csv(self.outpath / 'performance_metrics_test.csv', index=True)
            self.metrics_train.to_csv(self.outpath / 'performance_metrics_train.csv', index=True)


def plot_prediction(true_values: pd.Series | np.ndarray, predicted_values: pd.Series | np.ndarray, path: str | Path = None, it: int = 1):
    plt.figure()
    plt.scatter(predicted_values, true_values, c='b', marker='o')
    plt.plot([0, max(predicted_values)], [0, max(predicted_values)], c='r', label='ideal scenario')
    
    plt.ylabel(f"measured {true_values.name}")
    plt.xlabel(f"predicted {true_values.name}")
    plt.legend()
    if path:
        plt.savefig(path / f'estimation_{it}.png')
    plt.close()
