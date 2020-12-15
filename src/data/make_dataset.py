# -*- coding: utf-8 -*-
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pretty_errors
from dotenv import find_dotenv, load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from typeguard import typechecked

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"

sys.path.append(PROJECT_ROOT)

from src.base import get_settings
from src.base_pipeline import Extract, EPipeline
from src.make_logger import log_fun, make_logger
from tests.mock_dataset import mock_dataset


@typechecked
class Get_Raw_Data(BaseEstimator, TransformerMixin):
    """
    Load raw data

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Raises:
        NotImplementedError: Not ready to be used with SQL queries
        ValueError: when file extensions that are not sql, csv or parquet

    Returns:
        Get_Raw_Data: Instantiated object

    Example:
        >>> specs = {"float": [100, 1, 0.05] \
                    ,"int": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"bool": [100, 1, 0] \
                    ,"str": [100, 1, 0] \
                    ,"datetime": [100, 1, 0] \
                    }
        >>> df, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> basename = Path("data.csv")
        >>> path = PROJECT_ROOT / "data" / "raw"
        >>> df.to_csv(str(path / f"{basename}"), index=False)
        >>> grd = Get_Raw_Data(basename)
        >>> _ = grd.fit(meta_data)
        >>> loaded_df = grd.transform()
        >>> len(set(df.columns.values) - set(loaded_df.compute().columns.values)) == 0
        True
        >>> df.shape == loaded_df.compute().shape
        True
        >>> Path.unlink(path / f"{basename}")
    """
    @log_fun
    def __init__(self, basename: Path, path: Path=DATA / "raw"):
        self.basename = basename
        self.path = path

    @log_fun
    def fit(self, meta_data: pd.DataFrame, y=None):
        # Convert metadata

        # Due to nans, we need to read numerical data as float and boolean as object
        # TODO: Convert to int and float according to the nulls filter
        meta_data_dtypes_map = {"float": float, "int": float, "bool": "object", "str": str}

        mask = meta_data["python_dtype"].isin(list(meta_data_dtypes_map.keys()))
        meta_data.loc[mask, "python_dtype"] = meta_data.loc[mask, "python_dtype"].map(meta_data_dtypes_map)

        # Identify datetime columns
        mask_datetime = meta_data["python_dtype"] == "datetime64[ns]"
        self.datetime_columns = list(meta_data.loc[mask_datetime, "column_name"].values)

        # Create dict with column name and data type
        self.dtypes_mapping = pd.Series(meta_data.loc[~mask_datetime, "python_dtype"].values,
        index=meta_data.loc[~mask_datetime, "column_name"].values).to_dict()

        self.load_columns = meta_data["column_name"].values

        return self

    @log_fun
    def transform(self, X=None, y=None):
        if self.basename.suffix == ".csv":
            # Load data file
            data = dd.read_csv(str(self.path / self.basename), parse_dates=self.datetime_columns
                                ,date_parser=date_parser, dtype=self.dtypes_mapping)

        elif self.basename.suffix == ".parquet":
            data = dd.read_parquet(str(self.path / self.basename), 
                                    columns=self.load_columns)

        elif self.basename.suffix == ".sql":
            msg = "Read queries are not implemented yet."
            raise NotImplementedError(msg)

        else:
            msg = f"Wrong file format. Expected either: csv, parquet or sql. \
                    Got {Path(self.basename)}."
            raise ValueError(msg)

        return data
        
    @log_fun
    def get_feature_names(self):
        """
        Get lists of loaded columns

        Returns:
            (list): List of loaded columns
        """
        return self.load_columns


@typechecked
class Get_Meta_Data(BaseEstimator, TransformerMixin):
    """
    Load meta data

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Get_Meta_Data: Instantiated object

    Example:
        >>> specs = {"float": [100, 1, 0.05] \
                    ,"int": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"bool": [100, 1, 0] \
                    ,"str": [100, 1, 0] \
                    ,"datetime": [100, 1, 0] \
                    }
        >>> df, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> basename = Path("meta_mock.csv")
        >>> path = PROJECT_ROOT / "data" / "meta"
        >>> meta_data.to_csv(str(path / f"{basename}"), index=False)
        >>> gmd = Get_Meta_Data(basename, path)
        >>> _ = gmd.fit()
        >>> loaded_meta_data = gmd.transform()
        >>> meta_data.equals(loaded_meta_data)
        True
        >>> Path.unlink(path / f"{basename}")
    """
    @log_fun
    def __init__(self, basename: Path, path: Path=DATA / "meta"):
        self.basename = basename
        self.path = path

    @log_fun
    def fit(self, X=None, y=None):
        return self

    @log_fun
    def transform(self, X=None, y=None):
        return pd.read_csv(str(self.path / self.basename))
        
    @log_fun
    def get_feature_names(self):
        return None


@typechecked
class Split_Predictors_Target(BaseEstimator, TransformerMixin):
    # TODO: add tests and comments
    @log_fun
    def __init__(self):
        pass
    
    @log_fun
    def fit(self, meta_data: pd.DataFrame, y=None):
        mask_target = meta_data["is_model_target"] == True
        self.target = meta_data.loc[mask_target, "column_name"].values

        return self

    @log_fun
    def transform(self, data: dd, y=None):
        X = data[~self.target]
        y = data[self.target]
        return X, y
        
    @log_fun
    def get_feature_names(self):
        return None


@typechecked
class Split_Train_Test(BaseEstimator, TransformerMixin):
    # TODO: add tests and comments
    @log_fun
    def __init__(self, train_size: float):
        self.train_size = train_size
    
    @log_fun
    def fit(self, X=None, y=None):
        return self

    @log_fun
    def transform(self, X: dd, y: dd):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=self.train_size)

        return X_train, X_test, y_train, y_test
        
    @log_fun
    def get_feature_names(self):
        return None


@typechecked
class Split_Time(BaseEstimator, TransformerMixin):
    # TODO: add tests and comments
    @log_fun
    def __init__(self, split_date: str, time_dim_col: str):
        self.split_date = split_date
        self.time_dim_col = time_dim_col
    
    @log_fun
    def fit(self, X_train: dd, X_test: dd, y_train=None, y_test=None):
        self.mask_train_in_time = X_train[self.time_dim_col] <= self.split_date
        self.mask_test_in_time = X_test[self.time_dim_col] <= self.split_date

        return self

    @log_fun
    def transform(self, X_train: dd, X_test: dd, y_train=None, y_test=None):
        X = {"train": X_train[self.mask_train_in_time]
        ,"in-sample_out-time": X_train[~self.mask_train_in_time]
        ,"out-sample_in-time": X_test[self.mask_test_in_time]
        ,"out-sample_out-time": X_test[~self.mask_test_in_time]
        }

        y = {"train": y_train[self.mask_train_in_time]
            ,"in-sample_out-time": y_train[~self.mask_train_in_time]
            ,"out-sample_in-time": y_test[self.mask_test_in_time]
            ,"out-sample_out-time": y_test[~self.mask_test_in_time]
            }

        return X, y
        
    @log_fun
    def get_feature_names(self):
        return None


@log_fun
@typechecked
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


def make_get_data_steps(basename: Path) -> list:
    """
    Make the steps to be followed in the pipeline to read raw and meta data

    Args:
        basename (Path): Basename of the file to be read

    Returns:
        list: Steps to read raw data set

    Example:
        >>> # Mock a data set
        >>> specs = {"float": [100, 1, 0.05] \
                    ,"int": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"bool": [100, 1, 0] \
                    ,"str": [100, 1, 0] \
                    ,"datetime": [100, 1, 0] \
                    }
        >>> df, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> basename = Path("data.csv")
        >>> path = PROJECT_ROOT / "data"
        >>> meta_data.to_csv(str(path / "meta" / f"{basename}"), index=False)
        >>> df.to_csv(str(path / "raw" / f"{basename}"), index=False)
        >>> # Get the steps to get the data and instantiate the EPipeline
        >>> steps = make_get_data_steps(basename)
        >>> get_data_pipeline = EPipeline(steps)
        >>> # Fit and transform the pipeline
        >>> _ = get_data_pipeline.fit(None)
        >>> data = get_data_pipeline.transform(None)
        >>> # Test if raw data loaded by pipeline is what was expected
        >>> len(set(df.columns.values) - set(data.compute().columns.values)) == 0
        True
        >>> df.shape == data.compute().shape
        True
        >>> # Test if metada data loaded by pipeline is what was expected
        >>> loaded_meta_data = get_data_pipeline.named_steps['get_meta_data'].transform()
        >>> meta_data.equals(loaded_meta_data)
        True
        >>> # Delete mock data sets
        >>> Path.unlink(path / "raw" / f"{basename}")
        >>> Path.unlink(path / "meta" / f"{basename}")
    """

    # Read data steps
    return [("get_meta_data", Get_Meta_Data(basename))
            ,("get_raw_data", Get_Raw_Data(basename))
            ]


@click.command()
@click.argument('basename', type=click.Path())
@click.argument('save_interim', type=bool, default=True)
def main(basename, save_interim, from_interim):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Load

    ## Settings
    settings = get_settings()
    filter_thresholds = settings["thresholds"]
    train_size = settings["train"]["train_test_split_size"]
    train_size = settings["train"]["train_test_split_date"]
    split_date = settings["train"]["split_date"]
    time_dim_col = settings["features"]["time_dimension"]

    

    # TODO: Create Filter Data Steps

    # Splits
    split_steps = [
        ("split_predictors_target", Split_Predictors_Target())
        ,("split_train_test", Split_Train_Test(train_size))
        ,("split_time", Split_Time(split_date, time_dim_col))
    ]

if __name__ == '__main__':
    logger = make_logger(__file__)

    main()
