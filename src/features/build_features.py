# Use category_encoders
import os
import sys
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Union

from scipy import stats
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.base import BaseEstimator, TransformerMixin

SRC_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_PATH)
PROJECT_ROOT = Path(PROJECT_ROOT)
DATA = PROJECT_ROOT / "data"

sys.path.append(SRC_PATH)

CONTAINER_PREFIX = Path('/opt/ml/')
MODEL_PATH =  CONTAINER_PREFIX / 'model' if CONTAINER_PREFIX.is_dir() else PROJECT_ROOT / "models"

from base_pipeline import EPipeline, Extract, PandasFeatureUnion
from make_logger import log_fun


class Transform_Datetime(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self):
        pass
    
    @log_fun
    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names = copy(X.columns.values)
        return self

    @log_fun
    def transform(self, X: pd.DataFrame, y: pd.DataFrame=None):
        for column in self.feature_names:
            X.loc[:, f"{column}_year"] = X[column].dt.year
            X.loc[:, f"{column}_month"] = X[column].dt.month
            X.loc[:, f"{column}_day"] = X[column].dt.day

        self.feature_names = X.columns

        return X.drop(columns=self.feature_names)
        
    @log_fun
    def get_feature_names(self):
        return self.feature_names


class Input_Datetime(BaseEstimator, TransformerMixin):
    #! This is creating lots of nans. Do not use it.
    @log_fun
    def __init__(self, input_value: str='9999-12-31'):
        self.input_value = input_value
    
    @log_fun
    def fit(self, X: pd.DataFrame, y=None):
        self.columns = X.columns
        return self

    @log_fun
    def transform(self, X: pd.DataFrame, y=None):
        for column in self.columns:
            X[[column]].fillna(datetime.strptime(self.input_value, "%Y-%m-%d").date(), inplace=True)
            X[column] = pd.to_datetime(X[column])

        for column in X.columns:
            assert is_datetime(X[column]), f"Expected column {column} to be datetime. Got {X[column].dtypes}"
        return X
        
    @log_fun
    def get_feature_names(self):
        return self.feature_names


class Input_Numeric(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, method: Union[str, np.number]="median"):
        self.method = method
    
    @log_fun
    def fit(self, X: pd.DataFrame, y=None):
        self.col_input_val_map = {}
        self.feature_names = X.columns.values
        
        for column in X.columns.values:

            mask = X[column].notnull()
            array = X.loc[mask, column].values

            if self.method == "median":
                self.col_input_val_map[column] = np.quantile(array, 0.5)

            elif self.method == "25%":
                self.col_input_val_map[column] = np.quantile(array, 0.25)

            elif self.method == "mean":
                self.col_input_val_map[column] = np.mean(array)

            elif self.method == "min":
                self.col_input_val_map[column] = min(array)

            elif self.method == "max":
                self.col_input_val_map[column] = max(array)

            else:
                self.col_input_val_map[column] = self.method
        
        return self

    @log_fun
    def transform(self, X: pd.DataFrame, y=None):
        for column, value in self.col_input_val_map.items():
            X[column] = X[column].fillna(value)
        
        return X
        
    @log_fun
    def get_feature_names(self):
        return self.feature_names


@log_fun
def main(X: pd.DataFrame):

    # Encode numerical
    numerical_columns = X.select_dtypes(include=[np.number]).columns.values
    encode_numerical_steps = [("extract", Extract(column=numerical_columns))
                            ,("input_numeric", Input_Numeric())
                            ]
    numerical_encoder = EPipeline(encode_numerical_steps)
    # numerical_encoder.fit(X)

    # Encode datetime
    datetime_columns = X.select_dtypes(include=[np.datetime64]).columns.values
    encode_datetime_steps = [("extract", Extract(column=datetime_columns))
                            ,("transform_datetime", Transform_Datetime())]
    datetime_encoder = EPipeline(encode_datetime_steps)
    # datetime_encoder.fit(X)

    # Encode Categorical
    categorical_columns = X.select_dtypes(include=['category']).columns.values
    # TODO: Remove this np save and recover list of categorical from encoder object
    np.save(str(MODEL_PATH / "categorical_columns.npy"), categorical_columns)
    encode_categorical_steps = [("extract", Extract(column=categorical_columns))
                                # ,("categorize", Categorizer()) # N.B.: The categorize is necessary before running OneHotEncoder()
                                ,("transform_categorical", OneHotEncoder(handle_unknown="indicator", handle_missing="indicator"
                                                                        ,return_df=True, verbose=4))
                                ]
    categorical_encoder = EPipeline(encode_categorical_steps)
    # categorical_encoder.fit(X)

    encoder = PandasFeatureUnion([("numerical_encoder", numerical_encoder)
                            ,("datetime_encoder", datetime_encoder)
                            ,("categorical_encoder", categorical_encoder)
                            ])

    return encoder, categorical_columns
