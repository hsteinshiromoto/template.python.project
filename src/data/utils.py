import subprocess
import sys
from pathlib import Path
from typing import Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from typeguard import typechecked

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(PROJECT_ROOT)

# from tests.mock_dataset import mock_dataset
from src.make_logger import log_fun


@log_fun
@typechecked
def bin_and_agg(feature: str, data: pd.DataFrame, secondary_feature: str=None
                ,bins_boundaries: Union[np.array, str, bool]=True
                ,func: str="count"):
    # sourcery skip: remove-pass-elif
    """Aggregate feature according to bins. Use to Freedman-Diaconis Estimator 
    calculate bins [1].

    Args:
        feature (str): Feature binarized and aggregated, if a secondary_feature is not passed
        data (pd.DataFrame): Dataframe containing both features
        secondary_feature (str): Feature that is aggregated
        bins_boundaries (np.array or str or bool, optional): Array containing the bins. Defaults to True.
        func (str, optional): Function used to aggregate. Defaults to "count".

    Returns:
        pd.DataFrame: binarized and aggregated data

    References:
        [1] https://stats.stackexchange.com/questions/798/
            calculating-optimal-number-of-bins-in-a-histogram

    Example:
        >>> specs = {"float": [100, 1, 0] \
                        ,"int": [100, 1, 0] \
                        ,"categorical": [100, 1, 0] \
                        ,"bool": [100, 1, 0] \
                        ,"str": [100, 1, 0] \
                        ,"datetime": [100, 1, 0] \
                        }
        >>> data, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> df = bin_and_agg(feature="float_0", data=data)
        >>> "count_float_0" in df.columns.values
        True
        >>> df = bin_and_agg(feature="float_0", secondary_feature="int_0", data=data)
        >>> ("count_int_0" in df.columns.values) and ("count_float_0" not in df.columns.values)
        True
    """
    bin_edges_arg = ["auto", "fd", "doane", "scott", "stone", "rice"
                    , "sturges", "sqrt"]
    bin_time_freq = ["W", "M", "Q", "Y"]

    secondary_feature = secondary_feature or feature

    if bins_boundaries == True:
        bins_boundaries = np.histogram_bin_edges(data[feature].values, 
                                                bins="auto")

    elif bins_boundaries in bin_edges_arg:
        bins_boundaries = np.histogram_bin_edges(data[feature].values, 
                                                bins=bins_boundaries)

    elif not bins_boundaries:
        pass

    else:
        msg = f"Expected bins to be either {bin_edges_arg}, {bin_time_freq},\
                or bool. Got {bins_boundaries}."
        raise ValueError(msg)

    if isinstance(bins_boundaries, np.ndarray):
        groupby_args = pd.cut(data[feature], bins=bins_boundaries)

    elif bins_boundaries in bin_time_freq:
        data.set_index(feature, inplace=True)
        groupby_arg = pd.Grouper(key=feature, freq=bins_boundaries)

    else:
        groupby_args = feature

    grouped = data.groupby(groupby_args)[secondary_feature]

    return_dict = {"count": grouped.count
            ,"sum": grouped.sum
            ,"mean": grouped.mean
            ,"min": grouped.min
            ,"max": grouped.max
            }

    output = return_dict["count"]().to_frame(name=f"{func}_{secondary_feature}")
    output[f"cum_count_{secondary_feature}"] = output[f"count_{secondary_feature}"].cumsum()
    output[f"proportions_{secondary_feature}"] = 100.0*output[f"count_{secondary_feature}"]/output[f"count_{secondary_feature}"].sum()
    output[f"cum_proportions_{secondary_feature}"] = output[f"proportions_{secondary_feature}"].cumsum()

    if data[secondary_feature].dtype == np.number:
        output[f"mean_{secondary_feature}"] = return_dict["mean"]()
        output[f"min_{secondary_feature}"] = return_dict["min"]()
        output[f"max_{secondary_feature}"] = return_dict["max"]()

    return output


@log_fun
@typechecked
def aggregate_continuous(feature: str, data: pd.DataFrame
                        ,secondary_feature: str):
    """Aggregates continuous feature into bins and summarize statistics

    Args:
        feature (str): Feature binarized and aggregated.
        data (pd.DataFrame): Dataframe containing both features
        secondary_feature (str): Feature that is aggregated

    Returns:
        pd.DataFrame: binarized and aggregated data

    Example:
        >>> specs = {"float": [100, 1, 0] \
                        ,"int": [100, 1, 0] \
                        ,"categorical": [100, 1, 0] \
                        ,"bool": [100, 1, 0] \
                        ,"str": [100, 1, 0] \
                        ,"datetime": [100, 1, 0] \
                        }
        >>> data, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> df = aggregate_continuous(feature="float_0", data=data, secondary_feature="int_0")
    """

    summary = bin_and_agg(feature, data, secondary_feature=secondary_feature)
    summary[f"proportion_{secondary_feature}"] = 100.0*summary[f"count_{secondary_feature}"] / summary[f"count_{secondary_feature}"].sum()
    summary[f"mean_{secondary_feature}"] = bin_and_agg(feature, data, 
                        secondary_feature=secondary_feature, func="mean")

    summary[f"cum_count_{secondary_feature}"] = summary[f"count_{secondary_feature}"].cumsum()
    summary[f"cum_proportion_{secondary_feature}"] = 100.0 * summary[f"cum_count_{secondary_feature}"] / summary[f"count_{secondary_feature}"].sum()

    return summary


@log_fun
@typechecked
def aggregate_discrete(feature: str, data: pd.DataFrame
                        ,secondary_feature: str=None):
    """Aggregates continuous feature into bins and summarize statistics

    Args:
        feature (str): Feature binarized and aggregated.
        data (pd.DataFrame): Dataframe containing both features
        secondary_feature (str): Feature that is aggregated

    Returns:
        pd.DataFrame: binarized and aggregated data

    Example:
        >>> specs = {"float": [100, 1, 0] \
                        ,"int": [100, 1, 0] \
                        ,"categorical": [100, 1, 0] \
                        ,"bool": [100, 1, 0] \
                        ,"str": [100, 1, 0] \
                        ,"datetime": [100, 1, 0] \
                        }
        >>> data, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> df = aggregate_discrete(feature="float_0", data=data, secondary_feature="int_0")
    """

    secondary_feature = secondary_feature or feature

    return_dict = make_aggregate_dict(feature, secondary_feature, data)

    summary = return_dict["count"].to_frame(name=f"count_{secondary_feature}")
    summary[f"proportion_{secondary_feature}"] = 100.0*summary[f"count_{secondary_feature}"] / summary[f"count_{secondary_feature}"].sum()

    if data[secondary_feature].dtype == np.number:
        summary[f"mean_{secondary_feature}"] = return_dict["mean"]

    summary[f"cum_count_{secondary_feature}"] = summary[f"count_{secondary_feature}"].cumsum()
    summary[f"cum_proportion_{secondary_feature}"] = 100.0 * summary[f"cum_count_{secondary_feature}"] / summary[f"count_{secondary_feature}"].sum()

    return summary


@log_fun
@typechecked
def aggregate_time(feature: str, data: pd.DataFrame, freq="M"
                        ,secondary_feature: str=None):

    secondary_feature = secondary_feature or feature

    groupby_arg = pd.Grouper(key=feature, freq=freq)

    return_dict = make_aggregate_dict(groupby_arg, secondary_feature, data)

    summary = return_dict["count"].to_frame(name=f"count_{secondary_feature}")
    summary[f"proportion_{secondary_feature}"] = 100.0*summary[f"count_{secondary_feature}"] / summary[f"count_{secondary_feature}"].sum()

    if data[secondary_feature].dtype == np.number:
        summary[f"mean_{secondary_feature}"] = return_dict["mean"]

    summary[f"cum_count_{secondary_feature}"] = summary[f"count_{secondary_feature}"].cumsum()
    summary[f"cum_proportion_{secondary_feature}"] = 100.0 * summary[f"cum_count_{secondary_feature}"] / summary[f"count_{secondary_feature}"].sum()

    return summary