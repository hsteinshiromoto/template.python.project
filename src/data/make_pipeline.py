import subprocess
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

sys.path.append(str(PROJECT_ROOT))


class Extract(BaseEstimator, TransformerMixin):
    def __init__(self, column: list=None):
        self.column = column


    def fit(self, X: dd, y: dd=None):
        return X[self.column]


    def transform(self, X: dd, y: dd=None):
        return self
