import subprocess
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(str(PROJECT_ROOT))

from src.base_pipeline import Extract
from tests.mock_dataset import mock_dataset


def test_Extract():
    """
    Tests the Filter_Nulls pipeline
    """

    # Define a mock dataset in which the float column has 80% of values missing
    specs = {"float": [100, 1, 0.8]
            ,"integer": [100, 1, 0.025]
            ,"categorical": [100, 1, 0.1]
            ,"boolean": [100, 1, 0]
            ,"string": [100, 1, 0]
            }
    data = mock_dataset(specs)
    data = dd.from_pandas(data, npartitions=1)

    # Define what columns will be removed
    cols_to_be_selected = [col for col in data.columns.values if "float_" in col]

    # Instantiate the pipeline
    pipeline = Extract(cols_to_be_selected)
    selected_column_fit = pipeline.fit(data)
    selected_column_transform = pipeline.transform(data)

    # Test if the columns that would be removed were actually removed by the pipeline
    assert len(set(selected_column_fit.columns.values).symmetric_difference(selected_column_transform.columns.values)) == 0
    
    return


if __name__ == "__main__":
    pass