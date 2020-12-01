import subprocess
import sys
from datetime import datetime
from math import e, log
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))

from src.make_logger import log_fun, make_logger
from src.base_pipeline import EPipeline, Extract
from tests.mock_dataset import mock_dataset


class Filter_Nulls(BaseEstimator, TransformerMixin):
    """
    Filter columns according to the proportion of missing values

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Filter_Nulls: Instantiated object

    Example:
        >>> high = list(np.ones((50, 1)))
        >>> high.extend(50*[np.nan])
        >>> low = list(np.ones((99, 1)))
        >>> low.append(np.nan)
        >>> data = pd.DataFrame({"high": high, "low": low})
        >>> data = dd.from_pandas(data, npartitions=1)
        >>> filter_nulls = Filter_Nulls(0.3)
        >>> filter_nulls.fit(data)
        Filter_Nulls(nulls_threshold=0.3)
        >>> output = filter_nulls.transform(data)
        >>> "high" not in output.columns.values
        True
    """
    @log_fun
    def __init__(self, nulls_threshold: float=0.75):
        self.nulls_threshold = nulls_threshold

    @log_fun
    def fit(self, X: dd, y: dd=None):
        """
        Calculate what columns should be removed, based on the defined thresholds

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            None
        """

        # Calculate number of missing rows in each column
        summary_df = X.isnull().sum().compute()
        summary_df = summary_df.to_frame(name="nulls_count")
        summary_df["nulls_proportions"] = summary_df["nulls_count"] / X.shape[0].compute()
        summary_df.sort_values(by="nulls_count", ascending=False, inplace=True)

        # Select what columns should be removed, based on proportions
        mask_nulls = summary_df["nulls_proportions"] > self.nulls_threshold
        summary_df.loc[mask_nulls, "filtered_nulls"]  = 1
        summary_df.loc[~mask_nulls, "filtered_nulls"]  = 0
        
        self.feature_names = list(summary_df[mask_nulls].index.values)

        return self

    @log_fun
    def transform(self, X: dd, y: dd=None):
        """
        Remove columns computed in fit method

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            (dd): Dataframe with columns removed
        """
        return X.drop(labels=self.feature_names, axis=1)

    @log_fun
    def get_feature_names(self):
        """
        Get lists of removed columns

        Returns:
            (list): List of removed columns
        """
        return self.feature_names


class Filter_Std(BaseEstimator, TransformerMixin):
    """
    Filter columns according to the standard deviation of the columns

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Filter_Std: Instantiated object

    Example:
        >>> thresholds = [0.1, 1]
        >>> data = pd.DataFrame.from_dict({"low" : np.random.normal(0, 0.01, (100, 1)).squeeze() \
                                    ,"medium" : np.random.normal(0, np.mean(thresholds), (100, 1)).squeeze() \
                                    ,"high" : np.random.normal(0, 1.1, (100, 1)).squeeze() \
                                })
        >>> data = dd.from_pandas(data, npartitions=1)
        >>> filter_std = Filter_Std(thresholds)
        >>> filter_std.fit(data)
        Filter_Std(inclusive=False, std_thresholds=[0.1, 1])
        >>> output = filter_std.transform(data)
        >>> len(output.columns.values) == 1
        True
        >>> "medium" in output.columns.values
        True
    """
    @log_fun
    def __init__(self, std_thresholds: list=[0, np.inf], inclusive: bool=False):
        self.std_thresholds = std_thresholds
        self.inclusive = inclusive

    @log_fun
    def fit(self, X: dd, y: dd=None):
        """Calculate what columns should be removed, based on the defined thresholds

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            None
        """
        # Calculate the standad deviation column-wisely
        stds = np.nanstd(X, axis=0)

        stds_df = pd.DataFrame.from_dict({"column_name": X.columns.values
                                        ,"std": stds})

        stds_df.sort_values(by="std", inplace=True, ascending=False)

        # Get thresholds and calculate what columns will be removed
        thresholds = [float(value) for value in self.std_thresholds]
        mask_variance = stds_df["std"].between(min(thresholds), max(thresholds), inclusive=self.inclusive)

        # Get list of columns to be removed
        self.feature_names = list(stds_df.loc[~mask_variance, "column_name"].values)
        mask_removed = stds_df["column_name"].isin(self.feature_names)
        
        stds_df.loc[mask_removed, "filtered_variance"]  = 1
        stds_df.loc[~mask_removed, "filtered_variance"]  = 0
        
        return self

    @log_fun
    def transform(self, X: dd, y: dd=None):
        """
        Remove columns computed in fit method

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            (dd): Dataframe with columns removed
        """
        return X.drop(labels=self.feature_names, axis=1)

    @log_fun
    def get_feature_names(self):
        """
        Get lists of removed columns

        Returns:
            (list): List of removed columns
        """
        return self.feature_names


class Filter_Entropy(BaseEstimator, TransformerMixin):
    """
    Filter columns according to the entropy of the columns

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Filter_Std: Instantiated object

    Example:
        >>> thresholds = [0.1, 0.9]
        >>> medium = 50*["A"]
        >>> medium.extend(50*["B"])
        >>> high = []
        >>> for item in ["A", "B", "C", "D"]: \
            high.extend(25*[item])
        >>> data = pd.DataFrame.from_dict({"low": 100*["A"] \
                                        ,"medium": medium \
                                        ,"high": high \
                                        })
        >>> data = dd.from_pandas(data, npartitions=1)
        >>> filter_ent = Filter_Entropy(thresholds)
        >>> filter_ent.fit(data)
        Filter_Entropy(entropy_thresholds=[0.1, 0.9], inclusive=False)
        >>> output = filter_ent.transform(data)
        >>> len(output.columns.values) == 1
        True
        >>> "medium" in output.columns.values
        True
    """
    @log_fun
    def __init__(self, entropy_thresholds: list=[0, np.inf], 
                inclusive: bool=False):
        self.entropy_thresholds = entropy_thresholds
        self.inclusive = inclusive

    @log_fun
    def fit(self, X: dd, y: dd=None):
        """Calculate what columns should be removed, based on the defined thresholds

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            None
        """
        # Calculate the entropy column-wisely
        entropies_df = X.compute().apply(entropy, axis=0).to_frame(name="entropy")
        entropies_df.reset_index(inplace=True)
        entropies_df.rename(columns={"index": "column_name"}, inplace=True)
        entropies_df.sort_values(by="entropy", inplace=True, ascending=False)

        # Get thresholds and calculate what columns will be removed
        thresholds = [float(value) for value in self.entropy_thresholds]
        mask_entropy = entropies_df["entropy"].between(min(thresholds), max(thresholds), inclusive=self.inclusive)

        # Get list of columns to be removed
        self.feature_names = list(entropies_df.loc[~mask_entropy, "column_name"].values)
        mask_removed = entropies_df["column_name"].isin(self.feature_names)
        entropies_df.loc[mask_removed, "filtered_entropy"]  = 1

        return self

    @log_fun
    def transform(self, X: dd, y: dd=None):
        """
        Remove columns computed in fit method

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            (dd): Dataframe with columns removed
        """
        return X.drop(labels=self.feature_names, axis=1)

    @log_fun
    def get_feature_names(self):
        """
        Get lists of removed columns

        Returns:
            (list): List of removed columns
        """
        return self.feature_names


class Filter_Duplicates(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, subset: list=None):
        self.subset = subset

    @log_fun
    def fit(self, X: dd, y: dd=None):
        return self

    @log_fun
    def transform(self, X: dd, y: dd=None):
        """
        Remove duplicated rows

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            (dd): Dataframe with rows removed
        """
        return X.drop_duplicates(subset=self.subset)
    
    @log_fun
    def get_feature_names(self):
        """
        Get lists of removed columns

        Returns:
            (list): List of removed columns
        """
        return self.subset


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
                    ,pipeline: EPipeline=None, **kwargs) -> dd:
    """
    Creates filter pipeline

    Args:
        data (dd): Data to be filtered
        nulls (list or bool, optional): Columns to be filtered. Defaults to True.
        numerical (list or bool, optional): Columns to be filtered. Defaults to True.
        entropy (list or bool, optional): Columns to be filtered. Defaults to True.
        thresholds (dict, optional): Low and high thresholds. Defaults to {}.
        save_interim (bool, optional): Save filtered data to interim folder. Defaults to False.
        pipeline (EPipeline, optional): Built pipeline. Defaults to None.

    Returns:
        dd: [description]
    """
    if nulls:
        null_columns = data.columns.values if isinstance(nulls, bool) else nulls
        null_steps = [
        ("extract", Extract(null_columns))
        ,("filter_nulls", Filter_Nulls(thresholds.get("nulls")))
        ]
        null_pipeline = EPipeline(null_steps)
        pipeline = FeatureUnion([null_pipeline, pipeline]) if pipeline else null_pipeline

    if numerical:
        numerical_columns = data.select_dtypes(include=[np.number]).columns.values if isinstance(numerical, bool) else numerical
        numerical_steps = [
            ("extract", Extract(numerical_columns))
            ,("std_filter", Filter_Std(std_thresholds=thresholds.get("std")
            ,inclusive=kwargs.get("numerical")))
        ]
        std_filter_pipeline = EPipeline(steps=numerical_steps)
        pipeline = FeatureUnion([("std_filter_pipeline", std_filter_pipeline), ("existing_pipeline", pipeline)]) if pipeline else std_filter_pipeline

    if entropy:
        categorical_columns = data.select_dtypes(exclude=[np.number], include=["object"]) if isinstance(entropy, bool) else entropy
        categorical_steps = [
            ("extract", Extract(categorical_columns))
            ,("filter_variance", Filter_Entropy(entropy_thresholds=thresholds.get("std")
            ,inclusive=kwargs.get("entropy")))
        ]
        entropy_filter_pipeline = EPipeline(steps=categorical_steps)
        pipeline = FeatureUnion([("entropy_filter_pipeline", entropy_filter_pipeline), ("existing_pipeline", pipeline)]) if pipeline else entropy_filter_pipeline
        
    return pipeline
