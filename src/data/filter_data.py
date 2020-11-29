import subprocess
import sys
from datetime import datetime
from math import e, log
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))

from src.make_logger import log_fun, make_logger
from src.data.make_pipeline import Extract
from tests.mock_dataset import mock_dataset


class Filter_Nulls(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, nulls_threshold: float=0.75):
        self.nulls_threshold = nulls_threshold

    @log_fun
    def fit(self, X: dd, y: dd=None):
        summary_df = X.isnull().sum().compute()
        summary_df = summary_df.to_frame(name="nulls_count")
        summary_df["nulls_proportions"] = summary_df["nulls_count"] / X.shape[0].compute()
        summary_df.sort_values(by="nulls_count", ascending=False, inplace=True)

        mask_nulls = summary_df["nulls_proportions"] > self.nulls_threshold
        summary_df.loc[mask_nulls, "filtered_nulls"]  = 1
        summary_df.loc[~mask_nulls, "filtered_nulls"]  = 0
        
        self.removed_cols = list(summary_df[mask_nulls].index.values)

        return None

    @log_fun
    def transform(self, X: dd, y: dd=None):
        return X.drop(labels=self.removed_cols, axis=1)


class Filter_Std(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, std_thresholds: list=[0, np.inf], inclusive: bool=False):
        self.std_thresholds = std_thresholds
        self.inclusive = inclusive

    @log_fun
    def fit(self, X: dd, y: dd=None):
        stds = np.nanstd(X, axis=0)

        stds_df = pd.DataFrame.from_dict({"column_name": X.columns.values
                                        ,"std": stds})

        stds_df.sort_values(by="std", inplace=True, ascending=False)

        thresholds = [float(value) for value in self.std_thresholds]
        mask_variance = stds_df["std"].between(min(thresholds), max(thresholds), inclusive=self.inclusive)

        self.removed_cols = list(stds_df.loc[~mask_variance, "column_name"].values)
        mask_removed = stds_df["column_name"].isin(self.removed_cols)
        
        stds_df.loc[mask_removed, "filtered_variance"]  = 1
        stds_df.loc[~mask_removed, "filtered_variance"]  = 0
        
        return None

    @log_fun
    def transform(self, X: dd, y: dd=None):
        return X.drop(labels=self.removed_cols, axis=1)


class Filter_Entropy(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, entropy_thresholds: list=[0, np.inf], 
                inclusive: bool=False):
        self.entropy_thresholds = entropy_thresholds
        self.inclusive = inclusive

    @log_fun
    def fit(self, X: dd, y: dd=None):
        entropies_df = X.compute().apply(entropy, axis=0).to_frame(name="entropy")

        entropies_df.sort_values(by="entropy", inplace=True, ascending=False)

        thresholds = [float(value) for value in self.entropy_thresholds]
        mask_entropy = entropies_df["entropy"].between(min(thresholds), max(thresholds), inclusive=self.inclusive)
        self.removed_cols = list(entropies_df.loc[~mask_entropy, "column_name"].values)
        mask_removed = entropies_df["column_name"].isin(self.removed_cols)
        entropies_df.loc[mask_removed, "filtered_entropy"]  = 1

    return None

    @log_fun
    def transform(self, X: dd, y: dd=None):
        return X.drop(labels=self.removed_cols, axis=1)


class Filter_Duplicates(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, subset: list=None):
        self.subset = subset

    @log_fun
    def fit(self, X: dd, y: dd=None):
        return

    @log_fun
    def transform(self, X: dd, y: dd=None):
        return datXa.drop_duplicates(subset=self.subset)


@log_fun
def entropy(data, base: int=None) -> float:
    """
    Computes entropy of label distribution. 

    Args:
        data ([array-like]): Iterable containing data
        base ([int], optional): Base of logarithm to calculate entropy. Defaults to None.

    Returns:
        [float]: Information entropy of data

    Example:
        >>> entropy(np.ones(5))
        0

    References:
        [1] https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    """    

    n_labels = len(data)

    if n_labels <= 1:
        return 0

    _, counts = np.unique(data, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


@log_fun
def filter_pipeline(data: dd, nulls: list or bool=True
                    ,numerical: list or bool=True
                    ,entropy: list or bool=True
                    ,thresholds: dict={}, save_interim: bool=False
                    ,pipeline: Pipeline=None, **kwargs) -> dd:
    """
    Creates filter pipeline

    Args:
        data (dd): Data to be filtered
        nulls (list or bool, optional): Columns to be filtered. Defaults to True.
        numerical (list or bool, optional): Columns to be filtered. Defaults to True.
        entropy (list or bool, optional): Columns to be filtered. Defaults to True.
        thresholds (dict, optional): Low and high thresholds. Defaults to {}.
        save_interim (bool, optional): Save filtered data to interim folder. Defaults to False.
        pipeline (Pipeline, optional): Built pipeline. Defaults to None.

    Returns:
        dd: [description]
    """
    if nulls:
        null_columns = data.columns.values if isinstance(nulls, bool) else nulls
        null_steps = [
        ("extract", Extract(null_columns))
        ,("filter_nulls", Filter_Nulls(thresholds.get("nulls")))
        ]
        null_pipeline = Pipeline(null_steps)
        pipeline = FeatureUnion([null_pipeline, pipeline]) if pipeline else null_pipeline

    if numerical:
        numerical_columns = data.select_dtypes(include=[np.number]).columns.values if isinstance(numerical, bool) else numerical
        numerical_steps = [
            ("extract", Extract(numerical_columns))
            ,("filter_variance", Filter_Std(std_thresholds=thresholds.get("std")
            ,inclusive=kwargs.get("numerical")))
        ]
        numerical_pipeline = Pipeline(steps=numerical_steps)
        pipeline = FeatureUnion([numerical_pipeline, pipeline]) if pipeline else numerical_pipeline

    if entropy:
        categorical_columns = entropy if len(entropy) > 0 else \
                                data.select_dtypes(exclude=[np.number], include=["object"])
        categorical_steps = [
            ("extract", Extract(categorical_columns))
            ,("filter_variance", filter_entropy(data[categorical_columns]
            ,entropy_thresholds=thresholds.get("std")
            ,inclusive=kwargs.get("entropy")))
        ]
        categorical_pipeline = Pipeline(steps=categorical_steps)
        pipeline = FeatureUnion([categorical_pipeline, pipeline]) if pipeline else categorical_pipeline
        
    return pipeline