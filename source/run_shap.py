import os
# os.environ['TARGET'] = 'N_avg_app' # used for local testing
from data_loader import config_loader
from logging_util.logger import get_logger
from evaluation import feature_importance

logger = get_logger(__name__)
config = config_loader.load_config()

# apply shap values. Relevant parameters should be set as environment variables

if __name__ == '__main__':

    feature_importance.apply_shap(model_name=os.getenv('MODEL_NAME', 'HistGradientBoostRegressor'),
                                  fold=os.getenv('FOLD', 0),
                                  target=os.getenv('TARGET', 'N_avg_app'),
                                  test=True,
                                  subsample_n=int(os.getenv('SUBSAMPLE_N', 10)),
                                  resultspath=os.getenv('RESULTSPATH', ''))
