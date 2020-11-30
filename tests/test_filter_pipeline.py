import subprocess
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(str(PROJECT_ROOT))

from src.make_logger import log_fun, make_logger
from src.data.make_pipeline import Extract
from tests.mock_dataset import mock_dataset

def test_filter_pipeline():

    data = dd.from_pandas(mock_dataset(), npartitions=1)
    pipeline = filter_pipeline(data)

    assert isinstance(pipeline, Pipeline) or isinstance(pipeline, FeatureUnion)

    print(data)

    return None


if __name__ == "__main__":
    test_filter_pipeline()