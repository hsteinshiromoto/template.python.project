#!/usr/bin/env python3

import argparse
import subprocess
import pickle
import sys
from pathlib import Path
from typeguard import typechecked
from typing import Union
import pandas as pd
import scipy.stats as ss
import dask.dataframe as dd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost.sklearn import XGBClassifier

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"

sys.path.append(str(PROJECT_ROOT))

from src.base import Get_Settings, Get_Filename
from src.make_logger import log_fun, make_logger

@log_fun
class Get_Interim_Datasets(Get_Filename):
    def __init__(self, path: Path=PROJECT_ROOT / "data" / "interim"):
        super().__init__(path)

    def load(self, dataset_type: str):

        allowed_types = ["train", "in-sample_out-time", "test"
                        ,"out-sample_in-time", "out-sample_out-time"
                        ]
        if dataset_type not in allowed_types:
            msg = f"Expected dataset_type to be either {allowed_types}. Got {dataset_type}."
            raise ValueError(msg)
        
        X = {dataset_type: dd.read_parquet(str(self.path / f"X_{dataset_type}.parquet"))}
        y = {dataset_type: dd.read_parquet(str(self.path / f"y_{dataset_type}.parquet"))}

        return X, y


@log_fun
class Get_Pipeline(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, pipeline: str, timestamp: str=None
                ,path: Path=PROJECT_ROOT / "models"):
        self.timestamp = timestamp
        self.path = path
        self.pipeline = pipeline

    @log_fun
    def fit(self, X=None, y=None):
        if self.timestamp:
            self.basename = self.path / f"pipeline_{self.pipeline}_{self.timestamp}.pkl"

        else:
            self.basename = self.path / get_filename(self.path, newest=True, pattern=f"pipeline_{self.pipeline}_*.pkl")
        
        return self

    @log_fun
    def transform(self, X=None, y=None):
        with open(str(self.basename), 'rb') as input_file:
            pipeline = pickle.load(input_file)

        return pipeline

    @log_fun
    def get_feature_names(self):
        return None


@log_fun
def instantiate_estimator(objective='binary:logistic', use_label_encoder=False):
    return XGBClassifier(use_label_encoder=use_label_encoder, objective=objective)


@log_fun
def instantiate_cv(estimator, params_distr: dict, scale_pos_weight: float, n_iter: int=1
                    ,scoring=["roc_auc", "f1", "precision", "recall", "accuracy"]
                    ,cv=None, verbose: int=4, refit="roc_auc", n_splits: int=2
                    ,type: str=None):  
    # References:
        # [1] https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
        # [2] https://github.com/scikit-optimize/scikit-optimize/issues/1006

    # 2. Instantiate objects
    cv = cv or StratifiedKFold(n_splits=n_splits)

    if type == "bayes":
        # For future use. This is raising an error [2] for which I haven't found a solution yet
        raise NotImplementedError("Option Bayes Search CV not implemented")
        # distributions = make_bayessearch_cv_params_distr(params_distr, scale_pos_weight)
        # BayesSearchCV.__init__ = bayes_search_CV_init
        # cv_object = BayesSearchCV(estimator, search_spaces=distributions
        #                         ,scoring="roc_auc", cv=cv, verbose=verbose, refit=refit
        #                         ,n_iter=n_iter)

    distributions = make_randomsearch_cv_params_distr(params_distr, scale_pos_weight)
    return RandomizedSearchCV(
        estimator,
        param_distributions=distributions,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        refit=refit,
        n_iter=n_iter,
    )


@log_fun
def instantiate_callibrate_cv(estimator, method='sigmoid', cv="prefit"):
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


@log_fun
def make_randomsearch_cv_params_distr(estimator_parameters: dict, scale_pos_weight: float=None):

    distr_map = {"float":  {"normal": ss.norm, "loguniform": ss.loguniform}
                ,"int": {"uniform": ss.randint}
                }

    if scale_pos_weight:
        estimator_parameters["scale_pos_weight"]["values"][0] = scale_pos_weight
        estimator_parameters["scale_pos_weight"]["values"][1] *= scale_pos_weight

    params_dict = {}

    for parameter, properties in estimator_parameters.items():
        distribution_type = properties.get("distribution")
        data_type = properties["dtype"]
        values = properties["values"]

        if data_type == "str":
            params_dict[parameter] = values

        elif distribution_type in ["uniform", "normal"]:
            params_dict[parameter] = distr_map[data_type][distribution_type](*values)

    return params_dict


@log_fun
def process_data(data_loader: Get_Filename, basename: str):

    X_train = data_loader.load("")

    return X, y


@log_fun
def main(X: dict, y: dict, settings: dict=None, timestamp: str=None
        ,predict_steps: list=None):
    # References
    #    [1] https://github.com/dmlc/xgboost/issues/2334#issuecomment-406282203

    if not settings:
        settings = Get_Settings().load()

    # 2. Setup Training Parameters
    scale_pos_weight = y["train"].mean()

    # 3. Instantiate Estimator and Cross Validation Object
    estimator = instantiate_estimator()
    cv_settings = settings["cv"]
    params_distr = cv_settings["params_distr"]
    del cv_settings["params_distr"]
    
    cv = instantiate_cv(estimator, params_distr, scale_pos_weight, **cv_settings)

    # 4. Fit model and get best trained estimator
    # N.B. parse X_train as matrix to avoid feature bugs [1]

    cv.fit(X_train, y_train)
    estimator = cv.best_estimator_

    # 6. Save cross validation results
    cv_results_df = pd.DataFrame.from_dict(cv.cv_results_)
    cv_results_df.to_csv(str(METRICS_PATH / f"cv_results_{timestamp}.csv"), header=True, index=False)

    # 7. Save best estimator parameters
    estimator_params_df = pd.DataFrame(cv.best_params_, index=[0])
    estimator_params_df.to_csv(str(METRICS_PATH / f"uncalibrated_estimator_params_{timestamp}.csv"), header=True, index=False)

    calibrator_cv = instantiate_callibrate_cv(estimator)
    # N.B. parse X_train as matrix to avoid feature bugs [1]
    calibrator_cv.fit(X_cal, y_cal)

    estimators_dict = {"uncalibrated": estimator, "calibrated": calibrator_cv}

    with open(str(MODEL_PATH / f'estimators_dict_{timestamp}.pkl'), 'wb') as out:
        pickle.dump(estimators_dict, out)

    if predict_steps:
        pass

        #! Temporarily removed this part

        # predict_steps.append(("estimator", calibrator_cv))

        # with open(str(MODEL_PATH / f'predict_pipeline_{timestamp}.pkl'), 'wb') as out:
        #     print(f"Saving file predict_steps to {str(MODEL_PATH / f'predict_pipeline_{timestamp}.pkl')}")
        #     pickle.dump(predict_steps, out)
        #     print("Done.")

        # return estimators_dict, timestamp, predict_steps

    return estimators_dict, timestamp


if __name__ == '__main__':
    logger = make_logger(__file__)

    parser = argparse.ArgumentParser(description='Runs train_model')
    parser.add_argument('-t', '--timestamp', dest='timestamp', type=str, help='Timestamp')

    args = parser.parse_args()

    X, y = Get_Interim_Datasets().load("train")

    main(X, y)
