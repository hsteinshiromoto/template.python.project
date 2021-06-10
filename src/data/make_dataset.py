# -*- coding: utf-8 -*-
import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
# import pretty_errors
from icecream import ic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from typeguard import typechecked

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"

sys.path.append(str(PROJECT_ROOT))

from src.base import Get_Settings, argparse_str2bool
from src.base_pipeline import EPipeline, Extract
from src.data.filter_data import Filter_Entropy, Filter_Nulls, Filter_Std
from src.make_logger import log_fun, make_logger
from tests.mock_dataset import mock_dataset


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
            data = dd.read_parquet(str(self.path / self.basename))

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
class Predictors_Target_Split(BaseEstimator, TransformerMixin):
    """
    Splits data set into predictors and target

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Predictors_Target_Split: Instantiated object

    Example:
        >>> data = pd.DataFrame.from_dict({"predictor": np.random.rand(100, 1).flatten(), "target": np.random.rand(100, 1).flatten()})
        >>> pts = Predictors_Target_Split("target")
        >>> _ = pts.fit()
        >>> X, y = pts.transform(data)
        >>> ("target" not in X.columns.values) and ("predictor" in X.columns.values)
        True
        >>> ("predictor" not in y.columns.values) and ("target" in y.columns.values)
        True
    """
    @log_fun
    def __init__(self, y_col: str):
        self.y_col = y_col
    
    @log_fun
    def fit(self, X=None, y=None):
        return self

    @log_fun
    def transform(self, data: Union[dd.DataFrame, pd.DataFrame], y=None):
        X = data.loc[:, data.columns != self.y_col]
        y = data[[self.y_col]]

        self.predictors = X.columns.values
        return X, y
        
    @log_fun
    def get_feature_names(self):
        return self.predictors, self.y_col


@typechecked
class Train_Test_Split(BaseEstimator, TransformerMixin):
    """
    Splits train test sets

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Split_Predictors_Target: Instantiated object

    Example:
        >>> X, y = pd.DataFrame(np.random.rand(100, 1).flatten()), pd.DataFrame(np.random.randint(2, size=(100, 1)).flatten())
        >>> X, y = dd.from_pandas(X, npartitions=1), dd.from_pandas(y, npartitions=1)
        >>> tts = Train_Test_Split(0.75)
        >>> _ = tts.fit()
        >>> X_train, X_test, y_train, y_test = tts.transform(X, y)
        >>> X_train.shape[0].compute() > X_test.shape[0].compute()
        True
        >>> y_train.shape[0].compute() > y_test.shape[0].compute()
        True
    """
    @log_fun
    def __init__(self, train_proportion: float, n_splits: int=1):
        self.train_proportion = train_proportion
        self.n_splits = n_splits
    
    @log_fun
    def fit(self, X=None, y=None):
        
        self.sss = StratifiedShuffleSplit(n_splits=self.n_splits
                                        ,train_size=self.train_proportion)

        # N.B.: Need to add y to object to pass to .transform, when using in 
        # tandem with another pipe
        self.y = y

        return self

    @log_fun
    def transform(self, X: dd.DataFrame, y=None):
        n_partitions = X.npartitions

        # N.B.: Need to add y to object to pass to .transform, when using in 
        # tandem with another pipe
        if y is None:
            y = self.y

        for train_index, test_index in self.sss.split(X, y):
            X_train, X_test = X.compute().iloc[train_index, :], X.compute().iloc[test_index, :]
            y_train, y_test = y.compute().iloc[train_index, :], y.compute().iloc[test_index, :]

        X_dict, y_dict = {}, {}
        
        X_dict["train"] = dd.from_pandas(X_train, npartitions=n_partitions)
        X_dict["test"] = dd.from_pandas(X_test, npartitions=n_partitions)
        y_dict["train"] = dd.from_pandas(y_train, npartitions=n_partitions)
        y_dict["test"] = dd.from_pandas(y_test, npartitions=n_partitions)

        return X_dict, y_dict
        
    @log_fun
    def get_feature_names(self):
        return None


@typechecked
class Time_Split(BaseEstimator, TransformerMixin):
    """
    Splits data according to a certain date

    Args:
        BaseEstimator (BaseEstimator): Sci-kit learn object
        TransformerMixin (TransformerMixin): Sci-kit learn object

    Returns:
        Split_Predictors_Target: Instantiated object

    Example:
        >>> specs = {"float": [100, 1, 0.05] \
                    ,"int": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"bool": [100, 1, 0] \
                    ,"str": [100, 1, 0] \
                    ,"datetime": [100, 1, 0] \
                    }
        >>> df, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> spt = Split_Predictors_Target("str_0")
        >>> _ = spt.fit()
        >>> X, y = spt.transform(df)
        >>> tts = Train_Test_Split(0.75)
        >>> _ = tts.fit()
        >>> X_train, X_test, y_train, y_test = tts.transform(X, y)
        >>> ts = Time_Split(split_date=f"{df['datetime_0'].describe()['top'].date()}", time_dim_col="datetime_0")
        >>> _ = ts.fit(X_train, X_test)
        >>> X, y = ts.transform(X_train, X_test, y_train, y_test)
        >>> X["train"].shape[0] == y["train"].shape[0]
        True
        >>> X["in-sample_out-time"].shape[0] == y["in-sample_out-time"].shape[0]
        True
        >>> X["out-sample_in-time"].shape[0] == y["out-sample_in-time"].shape[0]
        True
        >>> X["out-sample_out-time"].shape[0] == y["out-sample_out-time"].shape[0]
        True
    """
    @log_fun
    def __init__(self, split_date: Union[str, datetime, datetime.date], time_dim_col: str):
        self.split_date = str(split_date)
        self.time_dim_col = time_dim_col
    
    @log_fun
    def fit(self, X: Union[dict, tuple], y=None):
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]

        self.mask_train_in_time = X["train"][self.time_dim_col] <= self.split_date
        self.mask_test_in_time = X["test"][self.time_dim_col] <= self.split_date

        self.y = y

        return self

    @log_fun
    def transform(self, X: Union[dict, tuple], y=None):
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
        
        if y is None:
            y = self.y

        n_partitions = X["train"].npartitions

        X_out = {"train": X["train"].loc[self.mask_train_in_time, :]
        ,"in-sample_out-time": X["train"].loc[~self.mask_train_in_time, :]
        ,"out-sample_in-time": X["test"].loc[self.mask_test_in_time, :]
        ,"out-sample_out-time": X["test"].loc[~self.mask_test_in_time, :]
        }

        y_out = {"train": y["train"].loc[self.mask_train_in_time, :]
            ,"in-sample_out-time": y["train"].loc[~self.mask_train_in_time, :]
            ,"out-sample_in-time": y["test"].loc[self.mask_test_in_time, :]
            ,"out-sample_out-time": y["test"].loc[~self.mask_test_in_time, :]
            }

        return X_out, y_out
        
    @log_fun
    def get_feature_names(self):
        return None


class Save_Dataset(BaseEstimator, TransformerMixin):
    @log_fun
    def __init__(self, basename: Path, path: Path):
        self.basename = basename
        self.path = path


    @log_fun
    def fit(self, X=None, y=None):
        return self


    @log_fun
    def transform(self, X=None, y=None):
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
            
        if self.basename.suffix == ".csv":
            # Load data file
            X.to_csv(str(self.path / self.basename.stem) + "_*.csv", index=False)
            y.to_csv(str(self.path / self.basename.stem) + "_*.csv", index=False)

        elif self.basename.suffix == ".parquet":
            dd.to_parquet(X
                        ,str(self.path / self.basename.stem) + "_*.parquet"
                        ,overwrite=True
                        )
            dd.to_parquet(y
                        ,str(self.path / self.basename.stem) + "_*.parquet"
                        ,overwrite=True
                        )

        else:
            msg = f"Wrong file format. Expected either: csv or parquet. \
                    Got {Path(self.basename)}."
            raise NotImplementedError(msg)


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


@log_fun
@typechecked
def get_data_steps(raw_data: Union[Path, str], meta_data: Union[Path,str]) -> list:
    """
    Make the steps to be followed in the pipeline to read raw and meta data

    Args:
        basename (Path): Basename of the file to be read

    Returns:
        list: Steps to read raw data set

    Example:
        # TODO: update unittests
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
    meta_data = Path(meta_data) if isinstance(meta_data, str) else meta_data
    raw_data = Path(raw_data) if isinstance(raw_data, str) else raw_data

    return [("get_meta_data", Get_Meta_Data(basename=meta_data))
            ,("get_raw_data", Get_Raw_Data(basename=raw_data))
            ]


@log_fun
@typechecked
def train_test_split_steps(y_col: str, train_proportion: float=0.75
                        ,time_split_settings: dict=None):
    """
    Make the steps split data set into training and test

    Args:
        settings (dict): Model build settings

    Returns:
        list: Steps for splitting the data set

    Example:
    # TODO: 
    """

    pred_target_split_pipe = EPipeline([("split_predictors_target", Predictors_Target_Split(y_col))])
    train_test_split_pipe = EPipeline([("split_train_test", Train_Test_Split(train_proportion))])

    if time_split_settings:
        split_date = time_split_settings["split_date"]
        time_dim_col = time_split_settings["time_dimension"]
        time_split_pipe = EPipeline([("split_time", Time_Split(split_date=split_date
                                    ,time_dim_col=time_dim_col)
                                    )])

    else:
        time_split_pipe = None

    return pred_target_split_pipe, train_test_split_pipe, time_split_pipe


@log_fun
@typechecked
def make_preprocess_steps(preprocess_settings: dict=None) -> list:

    return [("filter_nulls", Filter_Nulls())
            ,("filter_entropy", Filter_Entropy())
            ,("filter_std", Filter_Std())]


@typechecked
def main(data: Union[pd.DataFrame, dd.DataFrame, dict]=None, save: bool=False
        ,settings: dict={}, convert_to_parquet: bool=True):
    
    # Load

    ## Settings
    if not settings:
        settings = Get_Settings().load()

    if not data:
        data = settings["get_data"]

    if isinstance(data, dict):
        get_data_pipe = EPipeline(get_data_steps(**data))
        get_data_pipe.fit(None)
        data = get_data_pipe.transform(None)

    if settings["get_data"]["raw_data"].endswith(".csv") & convert_to_parquet:
        data.to_parquet(str(DATA / "raw" / f"data.parquet"))

    pred_target_split_pipe, train_test_split_pipe, time_split_pipe = train_test_split_steps(**settings["train_test_split"])
    pred_target_split_pipe.fit(None)
    X, y = pred_target_split_pipe.transform(data)

    train_test_split_pipe.fit(X, y)
    X, y = train_test_split_pipe.transform(X)

    if time_split_pipe is not None:
        time_split_pipe.fit(X, y)
        X, y = time_split_pipe.transform(X)

    for dataset_type in X.keys():
        X[dataset_type].to_parquet(str(DATA / "interim" / f"X_{dataset_type}.parquet"))
        y[dataset_type].to_parquet(str(DATA / "interim" / f"y_{dataset_type}.parquet"))

    return None


if __name__ == '__main__':
    logger = make_logger(__file__)

    # Create the parser
    parser = argparse.ArgumentParser(description='Runs make_dataset')
    parser.add_argument('-s', '--save', dest='save', type=argparse_str2bool
                        ,help='Save interim datasets', default=False)

    args = parser.parse_args()

    main(save=args.save)
