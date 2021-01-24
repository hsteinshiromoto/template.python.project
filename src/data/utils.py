import numpy as np
import pandas as pd
import dask.dataframe as dd

from typing import Union
import sys
import subprocess

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"

sys.path.append(PROJECT_ROOT)


from tests.mock_dataset import mock_dataset

def bin_and_agg(feature: str, bin_feature: str, data: pd.DataFrame
                ,bins: Union[np.array, str]=None, func: str="count"):
    """Aggregate feature according to bins

    Args:
        feature (str): Feature to be aggregated
        bin_feature (str): Feature that will be binarized
        data (pd.DataFrame): Dataframe containing both features
        bins (np.array or str, optional): Array containing the bins. Defaults to None.
        func (str, optional): Function used to aggregate. Defaults to "count".

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
        >>> df, meta_data = mock_dataset(specs=specs, meta_data=True)
        >>> 
    """

    # Use to Freedman Diaconis Estimator calculate bins
    # src: https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram

    if not bins:
        bins = np.histogram_bin_edges(data[bin_feature].values, bins="fd")

    return_dict = {"count": data.groupby(pd.cut(data[bin_feature], bins=bins))[feature].count()
                    ,"sum": data.groupby(pd.cut(data[bin_feature], bins=bins))[feature].sum()
                    ,"mean": data.groupby(pd.cut(data[bin_feature], bins=bins))[feature].mean()
                    ,"min": data.groupby(pd.cut(data[bin_feature], bins=bins))[feature].mean()
                    ,"max": data.groupby(pd.cut(data[bin_feature], bins=bins))[feature].mean()
                    }

    return return_dict[func].reset_index().to_frame(name=func)


def summarize_continuous(feature: str, data: pd.DataFrame):

    pass