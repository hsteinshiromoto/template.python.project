# -*- coding: utf-8 -*-
import logging
import subprocess
import sys
from pathlib import Path

import click
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"

sys.path.append(PROJECT_ROOT)

from src.base import get_settings
from src.make_logger import log_fun, make_logger


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

    value, counts = np.unique(data, return_counts=True)
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
def filter_nulls(data: dd, nulls_threshold: float=0.75):
    """
    Filter data set based on proportion of missing values

    Args:
        data (dask.dataframe): Data set to be filtered
        nulls_threshold (float, optional): Proportion of maximum number of missing values. Defaults to 0.75.

    Example:
        >>> data = dd.from_pandas(pd.DataFrame(np.random.rand(100, 20)), npartitions=1)
        >>> filtered_data, summary = filter_nulls(data)
        >>> isinstance(summary, pd.DataFrame)
        True
        >>> summary.shape[0] == 20
        True

    Returns:
        Union[dask.dataframe, pd.DataFrame]: Filtered data, summary of missing values
    """

    summary_df = data.isnull().sum().compute()
    summary_df = summary_df.to_frame(name="nulls_count")
    summary_df["nulls_proportions"] = summary_df["nulls_count"] / data.shape[0].compute()
    summary_df.sort_values(by="nulls_count", ascending=False, inplace=True)

    mask_nulls = summary_df["nulls_proportions"] > nulls_threshold
    summary_df.loc[mask_nulls, "filtered_nulls"]  = 1
    summary_df.loc[~mask_nulls, "filtered_nulls"]  = 0
    
    removed_cols = list(summary_df[mask_nulls].index.values)

    return data.drop(labels=removed_cols, axis=1), summary_df


@log_fun
def filter_numerical_variance(data: dd, std_thresholds: list=[0, np.inf], 
                                inclusive: bool=False):
    """
    Filter data set based on standard deviation of numerical values

    Args:
        data (dask.dataframe): Data set to be filtered
        variance_thresholds (list, optional): Standard deviation thresholds used to filter the data. Defaults to [0, np.inf].
        inclusive (bool, optional): Includes end points of the standard deviation thresholds. Defaults to False.

    Example:
        >>> data = dd.from_pandas(pd.DataFrame(np.random.rand(100, 20)), npartitions=1)
        >>> filtered_data, summary = filter_numerical_variance(data)
        >>> isinstance(summary, pd.DataFrame)
        True

    Returns:
        Union[dask.dataframe, pd.DataFrame]: Filtered data, summary of numerical columns
    """
    summary_df = data.select_dtypes(include=[np.number]).describe().compute()
    summary_df = summary_df.T.reset_index()
    summary_df.rename(columns={"index": "column_name"}, inplace=True)
    summary_df.sort_values(by="column_name", inplace=True)

    thresholds = [float(value) for value in std_thresholds]
    mask_variance = summary_df["std"].between(min(thresholds), max(thresholds), inclusive=inclusive)

    removed_cols = list(summary_df.loc[~mask_variance, "column_name"].values)
    mask_removed = summary_df["column_name"].isin(removed_cols)
    
    summary_df.loc[mask_removed, "filtered_variance"]  = 1
    summary_df.loc[~mask_removed, "filtered_variance"]  = 0
    
    return data.drop(labels=removed_cols, axis=1), summary_df.set_index("column_name")


@log_fun
def filter_entropy(data: dd, entropy_thresholds: list=[0, np.inf],
                    inclusive: bool=False):
    """
    Filter data set based on entropy

    Args:
        data (dask.dataframe): Data set to be filtered
        entropy_thresholds (list, optional): Entropy value thresholds used to filter the data. Defaults to [0, np.inf].
        inclusive (bool, optional): Includes end points of the standard deviation thresholds. Defaults to False.

    Returns:
        Union[dask.dataframe, pd.DataFrame]: Filtered data, summary of numerical columns
    """

    summary_df = data.select_dtypes(exclude=[np.number], include=["object"]).describe().compute()
    summary_df = summary_df.T

    entropies = data.select_dtypes(exclude=[np.number], include=["object"]).compute().apply(entropy, axis=0)
    entropies = entropies.to_frame(name="entropy")

    summary_df = summary_df.merge(entropies, left_index=True, right_index=True)

    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "column_name"}, inplace=True)
    summary_df.sort_values(by="column_name", inplace=True)

    thresholds = [float(value) for value in entropy_thresholds]
    mask_entropy = summary_df["entropy"].between(min(thresholds), max(thresholds), inclusive=inclusive)
    removed_cols = list(summary_df.loc[~mask_entropy, "column_name"].values)
    mask_removed = summary_df["column_name"].isin(removed_cols)
    summary_df.loc[mask_removed, "filtered_entropy"]  = 1
    summary_df.loc[~mask_removed, "filtered_entropy"]  = 0
    
    return data.drop(labels=removed_cols, axis=1), summary_df.set_index("column_name")


@log_fun
def date_parser(array, format: str="%Y-%m-%d"):
    """Converts array with dates to datetime format

    Args:
        array (hashable): Array containing dates and/or time
        format (str, optional): Datetime format. Defaults to "YYYY-MM-DD".

    Returns:
        [type]: Array with elements transformed into datetime format

    Example:
        >>> array = np.array(["2020-01-20", "2019-10-01"])
        >>> output = date_parser(array)
        >>> np.issubdtype(output.values.dtype, np.datetime64)
        True
    """

    return pd.to_datetime(array, format=format)


def get_raw_data(basename: str, meta_data: pd.DataFrame, path: Path=DATA / "raw"):
    """
    Reads raw data

    Args:
        basename (str): Filename
        meta_data (pd.DataFrame): Data frame describing the raw data
        path (Path, optional): Path to raw data. Defaults to DATA/"raw".

    Raises:
        NotImplementedError: SQL queries are not yet implemented
        ValueError: Only csv, parquet, and SQL are accepted at the moment

    Returns:
        dask.dataframe: Raw dataframe
    """

    try:
        ignore_mask = meta_data["ignore"] == True

    except KeyError:
        ignore_mask = [True for i in range(meta_data.shape[0])]

    meta_data = meta_data[ignore_mask]

    if basename.endswith("csv"):
    
        # Identify datetime columns
        mask_datetime = meta_data["python_dtypes"] == "datetime64[ns]"
        datetime_columns = list(meta_data[mask_datetime, "python_dtypes"].values)

        # Create dict with column name and data type
        dtypes_mapping = {zip(meta_data.loc[~mask_datetime, "column_name"].values, 
                        meta_data.loc[~mask_datetime, "python_dtypes"].values)}

        # Load data file
        data = dd.read_csv(str(path / basename), parse_dates=datetime_columns
                            ,date_parser=date_parser, dtypes=dtypes_mapping)

    elif basename.endswith("parquet"):
        data = dd.read_parquet(str(path / basename), 
                                columns=meta_data["columns_name"].values)

    elif basename.endswith("sql"):
        msg = "Read queries are not implemented yet."
        raise NotImplementedError(msg)

    else:
        msg = f"Wrong file format. Expected either: csv, parquet or sql. \
                Got {Path(basename)}."
        raise ValueError(msg)

    return data


@click.command()
@click.argument('basename', type=str)
@click.argument('output_filepath', type=click.Path())
def main(basename, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    # Load Settings

    # Load Metadata
    meta_data = pd.read_csv(str(DATA / "meta" / f"{basename}"))

    # Load Raw Data
    raw_data = get_raw_data(basename, meta_data)



if __name__ == '__main__':
    logger = make_logger(__file__)

    main()
