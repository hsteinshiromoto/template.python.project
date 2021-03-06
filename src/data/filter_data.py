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
from typeguard import typechecked

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))

from src.base_pipeline import EPipeline, Extract
from src.make_logger import log_fun, make_logger
from tests.mock_dataset import mock_dataset


@typechecked
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
        >>> _ = filter_nulls.fit(data)
        >>> output = filter_nulls.transform(data)
        >>> "high" not in output.columns.values
        True
    """
    @log_fun
    def __init__(self, nulls_threshold: float=0.75):
        self.nulls_threshold = nulls_threshold

    @log_fun
    def fit(self, X: dd, y=None):
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
    def transform(self, X: dd, y=None):
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


@typechecked
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
        >>> _ = filter_std.fit(data)
        >>> output = filter_std.transform(data)
        >>> (output.shape[1] == 1) & ("medium" in output.columns.values)
        True
    """
    @log_fun
    def __init__(self, std_thresholds: list[float]=[0, np.inf], inclusive: bool=False):
        self.std_thresholds = std_thresholds
        self.inclusive = inclusive

    @log_fun
    def fit(self, X: dd, y=None):
        """Calculate what columns should be removed, based on the defined thresholds

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            None
        """
        subset = X.select_dtypes(include=[np.number])

        # Calculate the standad deviation column-wisely
        stds = np.nanstd(subset, axis=0)

        stds_df = pd.DataFrame.from_dict({"column_name": subset.columns.values
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
    def transform(self, X: dd, y=None):
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


@typechecked
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
        >>> _ = filter_ent.fit(data)
        >>> output = filter_ent.transform(data)
        >>> (output.shape[1] == 1)
        True
        >>> "medium" in output.columns.values
        True
    """
    @log_fun
    def __init__(self, entropy_thresholds: list[float]=[0, np.inf], 
                inclusive: bool=False):
        self.entropy_thresholds = entropy_thresholds
        self.inclusive = inclusive

    @log_fun
    def fit(self, X: dd, y=None):
        """Calculate what columns should be removed, based on the defined thresholds

        Args:
            X (dd): Dataframe to be processed
            y (dd, optional): Target. Defaults to None.

        Returns:
            None
        """
        subset = X.select_dtypes(exclude=[np.number, "datetime64[ns]"])

        # Calculate the entropy column-wisely
        entropies_df = subset.compute().apply(entropy, axis=0).to_frame(name="entropy")
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
    def transform(self, X: dd, y=None):
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


@typechecked
class Filter_Duplicates(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, subset: list[str]=None):
        self.subset = subset

    @log_fun
    def fit(self, X: dd, y=None):
        return self

    @log_fun
    def transform(self, X: dd, y=None):
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


@typechecked
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


@typechecked
@log_fun
def make_filter_nulls_pipeline(data: dd, null_columns: list[str] or bool=True
                                ,threshold: float=None):
    """
    Makes pipeline to filter columns according to missing values

    Args:
        data (dd): Data frame to be filtered
        null_columns (listorbool, optional): Columns to subset the filtering. Defaults to True.
        threshold (float, optional): Maximum proportion of missing values missing. Defaults to None.

    Returns:
        EPipeline: Pipeline to filter data frame

    Example:
        >>> high = list(np.ones((50, 1)))
        >>> high.extend(50*[np.nan])
        >>> low = list(np.ones((99, 1)))
        >>> low.append(np.nan)
        >>> remaining_col = list(np.ones((100, 1)))
        >>> data = pd.DataFrame({"high": high, "low": low, "remain": remaining_col})
        >>> data = dd.from_pandas(data, npartitions=1)
        >>> filter_nulls = make_filter_nulls_pipeline(data, null_columns=["high", "low"], threshold=0.3)
        >>> _ = filter_nulls.fit(data)
        >>> output = filter_nulls.transform(data)
        >>> "high" not in output.columns.values
        True
        >>> len(output.columns.values) == 1
        True
    """

    selected_columns = data.columns.values if isinstance(null_columns, bool) else null_columns
    steps = [("extract", Extract(selected_columns))
            ,("filter_nulls", Filter_Nulls(threshold))
            ]
    
    return EPipeline(steps)


@typechecked
@log_fun
def make_filter_std_pipeline(data: dd, numerical_columns: list[str] or bool=True
                            ,thresholds: list[float]=None, inclusive: bool=False):
    #TODO: write unit tests
    """
    Makes pipeline to filter columns according to standard deviation

    Args:
        data (dd): Data frame to be filtered
        numerical_columns (list or bool, optional): Columns to subset the filtering. Defaults to True.
        thresholds (list, optional): Interval of std values to filter. Defaults to None.
        inclusive (bool, optional):  Includes or not the interval boundaries. Defaults to False.

    Returns:
        EPipeline: Pipeline to filter data frame
    """
    selected_columns = data.select_dtypes(include=[np.number]).columns.values if isinstance(numerical_columns, bool) else numerical_columns
    steps = [("extract", Extract(selected_columns))
        ,("std_filter", Filter_Std(std_thresholds=thresholds, inclusive=inclusive))
            ]

    return EPipeline(steps)


@typechecked
@log_fun
def make_filter_entropy_pipeline(data: dd, categorical_columns: list[str] or bool=True
                                ,thresholds: list[float]=None, inclusive: bool=False):
    #TODO: write unit tests
    selected_columns = data.select_dtypes(exclude=[np.number], include=["object"]) if isinstance(categorical_columns, bool) else categorical_columns
    steps = [("extract", Extract(selected_columns))
            ,("entropy_filter", Filter_Entropy(entropy_thresholds=thresholds
                                                ,inclusive=inclusive))
            ]

    return EPipeline(steps)


@typechecked
@log_fun
def filter_pipeline(data: dd, null_columns: list[str] or bool=True
                    ,numerical_columns: list[str] or bool=True
                    ,categorical_columns: list[str] or bool=True
                    ,thresholds: dict={}, save_interim: bool=False
                    ,pipeline: EPipeline=None, **kwargs) -> dd:
    #TODO: write unit tests
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
    pipe_dict = {}

    if null_columns:
        pipe_dict["nulls_pipeline"] = make_filter_nulls_pipeline(data, null_columns=null_columns, threshold=thresholds.get("nulls"))

    if numerical_columns:
        pipe_dict["std_pipeline"] = make_filter_std_pipeline(data, numerical_columns=numerical_columns, thresholds=thresholds.get("numerical"), inclusive=kwargs.get("numerical"))

    if categorical_columns:
        pipe_dict["entropy_pipeline"] = make_filter_entropy_pipeline(data, categorical_columns=categorical_columns, thresholds=thresholds.get("entropy"), inclusive=kwargs.get("entropy"))
        
    return pipe_dict
