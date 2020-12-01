import subprocess
import sys
from pathlib import Path
import numpy as np

import dask.dataframe as dd
import pandas as pd
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(str(PROJECT_ROOT))

from src.data.filter_data import Filter_Nulls, Filter_Std, Filter_Entropy, filter_pipeline
from src.base_pipeline import EPipeline, Extract
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
    pipeline = EPipeline([("filter_nulls", Filter_Nulls())])
    pipeline.fit(data)

    # Get the name of columns that have been removed
    removed_columns = pipeline.get_feature_names()

    # Process the dataframe
    output = pipeline.transform(data)

    # Get set of names of columns that were note removed
    cols_not_removed = set(data.columns.values) - set(list(removed_columns.values())[0])

    # Test if the columns that would be removed were actually removed by the pipeline
    assert len(set(cols_to_be_removed).symmetric_difference(list(removed_columns.values())[0])) == 0
    
    # Test if the columns that should not be removed were not actually removed
    assert len(cols_not_removed - set(output.columns.values)) == 0
    
    return


def test_filter_std():
    """
    Tests the Filter_Std class in a EPipeline

    Returns:
        None
    """

    specs = {"float": [100, 1, 0.8]
        ,"integer": [100, 1, 0.025]
        ,"categorical": [100, 1, 0.1]
        ,"boolean": [100, 1, 0]
        ,"string": [100, 1, 0]
            }

    # Test 1: No columns are removed
    mock_data = mock_dataset(specs)
    mock_data.drop(columns=[col for col in mock_data.columns.values if "float_" in col], inplace=True)

    std = 1

    data = pd.DataFrame.from_dict({"float" : list(np.random.normal(0, std, (mock_data.shape[0], 1)).squeeze())})
    data = data.merge(mock_data, left_index=True, right_index=True)
    data = dd.from_pandas(data, npartitions=1)

    # Instantiate the pipeline
    pipeline = Filter_Std()
    pipeline.fit(data.select_dtypes(include=[np.number]))
    output = pipeline.transform(data.select_dtypes(include=[np.number]))

    # Get the name of columns that have been removed
    removed_columns = pipeline.get_feature_names()

    assert not removed_columns

    assert len(set(data.select_dtypes(include=[np.number]).columns.values) - set(output.columns.values)) == 0

    # Test 2: The Float_0 column is removed

    ## Make the dataset
    mock_data = mock_dataset(specs)
    mock_data.drop(columns=[col for col in mock_data.columns.values if "float_" in col], inplace=True)

    thresholds = [0.1, 1]

    data = pd.DataFrame.from_dict({"float_0" : list(np.random.normal(0, np.mean(thresholds), (mock_data.shape[0], 1)).squeeze())
                                    ,"float_1" : list(np.random.normal(0, 0.01, (mock_data.shape[0], 1)).squeeze())
                                    ,"float_2" : list(np.random.normal(0, 1.1, (mock_data.shape[0], 1)).squeeze())
                                })
    data = data.merge(mock_data, left_index=True, right_index=True)
    data = dd.from_pandas(data, npartitions=1)

    ## Columns that shall be removed and remain
    cols_to_be_removed = ["float_1", "float_2"]
    cols_to_remain = ["float_0"]

    ## Create steps for pipeline: select float columns and filter
    steps = [("extract", Extract(["float_0", "float_1", "float_2"]))
            ,("filter_std", Filter_Std(thresholds))
            ]
    pipeline = EPipeline(steps=steps)
    pipeline.fit(data)
    output = pipeline.transform(data)

    # Get the name of columns that have been removed by Filter_Std method
    removed_columns = pipeline.get_feature_names()["filter_std"]

    # Process the dataframe
    output = pipeline.transform(data)

    # Test if the columns that would be removed were actually removed by the pipeline
    assert len(set(cols_to_be_removed).symmetric_difference(removed_columns)) == 0

    # Test if the columns that should remain were not removed
    assert len(set(cols_to_remain) - set(output.columns.values)) == 0

    return None


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

    # Define what columns will be selected.
    # N.B. the float columns will be removed
    selected_cols = [col for col in data.columns.values if "float_" in col]
    selected_cols.extend([col for col in data.columns.values if "integer_" in col])

    # Define the pipeline steps
    null_steps = [("extract", Extract(selected_cols))
                ,("filter_nulls", Filter_Nulls())
                ]

    # Instantiate the pipeline object
    pipeline = Pipeline(null_steps)

    # Fit and transform
    pipeline.fit(data)
    output = pipeline.transform(data)

    # Test if the integer columns are in the data frame
    assert len([col for col in output.columns.values if "integer_" in col]) > 0

    # Test if the float columns have been removed
    assert len([col for col in output.columns.values if "float_" in col]) == 0

    return None

def test_filter_pipeline():

    data = dd.from_pandas(mock_dataset(), npartitions=1)
    pipeline = filter_pipeline(data)

    assert isinstance(pipeline, Pipeline) or isinstance(pipeline, FeatureUnion)

    print(data)

    return None


if __name__ == "__main__":
    test_nulls_composition()