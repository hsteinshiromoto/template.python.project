import subprocess
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(str(PROJECT_ROOT))


class EPipeline(Pipeline):
    """
    Extends Pipeline class to add the get_feature_names method

    Args:
        Pipeline (Pipeline): Sci-kit learn pipeline class

    Src:
        https://stackoverflow.com/questions/48005889/get-features-from-sklearn-feature-union
    """
    def __init__(self, steps: list, **kwargs):
        super().__init__(steps=steps, **kwargs)

    def get_feature_names(self):
        return {name: step.get_feature_names() for name, step in self.steps}


class Extract(BaseEstimator, TransformerMixin):
    """
    Extracts specific columns of a data frame

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn BaseEstimator class
        TransformerMixin (TransformerMixin): Sci-kit learn TransformerMixin class
    """
    def __init__(self, column: list=None):
        if not isinstance(column, list):
            msg = f"Expected argument to be of type list. Got {type(column)}."
            raise TypeError(msg)
        self.column = column

    def fit(self, X: dd, y: dd=None):
        return self

    def transform(self, X: dd, y: dd=None):
        return X[self.column]

    def get_feature_names(self):
        return self.column