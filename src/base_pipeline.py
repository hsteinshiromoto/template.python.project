import subprocess
import sys
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import (FeatureUnion, Pipeline, _fit_transform_one,
                              _transform_one)

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
    def __init__(self, column: list[str]=None):
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


class PandasFeatureUnion(FeatureUnion):
    """Feature Union return Dataframes

    Args:
        (FeatureUnion): [description]

    References:
        [1] https://github.com/marrrcin/pandas-feature-union/blob/master/pandas_feature_union.py
        [2] https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html
    """
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
