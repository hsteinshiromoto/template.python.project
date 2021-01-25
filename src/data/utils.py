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
DATA = PROJECT_ROOT / "data"

sys.path.append(PROJECT_ROOT)

from tests.mock_dataset import mock_dataset


@typechecked
def bin_and_agg(feature: str, data: pd.DataFrame, secondary_feature: str=None
                ,bins: Union[np.array, str]=None, func: str="count"):
    """Aggregate feature according to bins. Use to Freedman-Diaconis Estimator 
    calculate bins [1].

    Args:
        feature (str): Feature binarized and aggregated, if a secondary_feature is not passed
        data (pd.DataFrame): Dataframe containing both features
        secondary_feature (str): Feature that is aggregated
        bins (np.array or str, optional): Array containing the bins. Defaults to None.
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
    
    bins = bins or np.histogram_bin_edges(data[feature].values, bins="fd")

    secondary_feature = secondary_feature or feature

    return_dict = {"count": data.groupby(pd.cut(data[feature], bins=bins))[secondary_feature].count()
                    ,"sum": data.groupby(pd.cut(data[feature], bins=bins))[secondary_feature].sum()
                    ,"mean": data.groupby(pd.cut(data[feature], bins=bins))[secondary_feature].mean()
                    ,"min": data.groupby(pd.cut(data[feature], bins=bins))[secondary_feature].min()
                    ,"max": data.groupby(pd.cut(data[feature], bins=bins))[secondary_feature].max()
                    }

    return return_dict[func].to_frame(name=f"{func}_{secondary_feature}").\
            reset_index().rename(columns={feature: "bins"})
