import subprocess
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(str(PROJECT_ROOT))

from src.data.filter_data import Filter_Nulls, Filter_Std, Filter_Entropy, filter_pipeline
from src.base_pipeline import Extract
from tests.mock_dataset import mock_dataset


def test_filter_nulls():
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
    cols_to_be_removed = [col for col in data.columns.values if "float_" in col]

    # Instantiate the pipeline
    pipeline = Filter_Nulls()
    pipeline.fit(data)

    # Get the name of columns that have been removed
    removed_columns = pipeline.get_removed_columns()

    # Process the dataframe
    output = pipeline.transform(data)

    # Get set of names of columns that were note removed
    cols_not_removed = set(data.columns.values) - set(removed_columns)

    # Test if the columns that would be removed were actually removed by the pipeline
    assert len(set(cols_to_be_removed).symmetric_difference(removed_columns)) == 0
    
    # Test if the columns that should not be removed were not actually removed
    assert len(cols_not_removed - set(output.columns.values)) == 0
    
    return


def test_filter_std():
    pass


def test_filter_entropy():
    pass


def test_nulls_composition():
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
    cols_to_be_removed = [col for col in data.columns.values if "float_" in col]
    cols_to_be_removed.extend([col for col in data.columns.values if "integer_" in col])

    # Instantiate the pipeline
    null_steps = [("extract", Extract(cols_to_be_removed))
                ,("filter_nulls", Filter_Nulls())
                ]
    pipeline = Pipeline(null_steps)

    pipeline.fit(data)
    output = pipeline.transform(data)
    print(output)

    pass

def test_filter_pipeline():

    data = dd.from_pandas(mock_dataset(), npartitions=1)
    pipeline = filter_pipeline(data)

    assert isinstance(pipeline, Pipeline) or isinstance(pipeline, FeatureUnion)

    print(data)

    return None


if __name__ == "__main__":
    test_nulls_composition()