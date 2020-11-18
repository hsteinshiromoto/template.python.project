import pandas as pd
import numpy as np
from pprint import pprint

def mock_dataset(specs: dict=None,):
    """
    Create mock pandas dataframe

    Args:
        specs (dict, optional): Specifications of the data frame. Defaults to None.

    Returns:
        pd.DataFrame: mock pandas dataframe

    Example:
        >>> specs = {"float": [100, 1, 0.05] \
                    ,"integer": [100, 1, 0.025] \
                    ,"categorical": [100, 1, 0.1] \
                    ,"boolean": [100, 1, 0] \
                    ,"string": [100, 1, 0] \
                    }
        >>> df = mock_dataset(specs)
        >>> df.shape[0] == 100
        True
        >>> df.shape[1] == 5
        True
        >>> df.isnull().sum()["float_0"] / df.shape[0] == 0.05
        True
        >>> df.isnull().sum()["integer_0"] / df.shape[0] == 0.02
        True
        >>> df.isnull().sum()["categorical_0"] / df.shape[0] == 0.1
        True
        >>> df.isnull().sum()["boolean_0"] / df.shape[0] == 0
        True
        >>> df.isnull().sum()["string_0"] / df.shape[0] == 0
        True
    """

    # 1. Build specs, in case needed
    if not specs:
        # Format of specs dict: {data_type: [nrows, ncols, nnulls]}
        specs = {"float": [100, np.random.randint(1, 4), np.random.rand()]
                ,"integer": [100, np.random.randint(1, 4), np.random.rand()]
                ,"categorical": [100, np.random.randint(1, 4), 0.75]
                ,"boolean": [100, np.random.randint(1, 4), np.random.rand()]
                ,"string": [100, np.random.randint(1, 4), np.random.rand()]
                }

    # 2. Build values of the data frame
    values = {}
    for col_type, col_spec in specs.items():
        if col_type == "float":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = np.random.rand(col_spec[0], 1).flatten()

        elif col_type == "integer":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = np.random.randint(np.random.randint(1e6), size=col_spec[0]).flatten()

        elif col_type == "categorical":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = ["".join(category.flatten()) for category in np.random.choice(["A", "B", "C", "D"], size=(col_spec[0], 3))]

        elif col_type == "boolean":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = [bool(item) for item in np.random.randint(2, size=col_spec[0])]

        elif col_type == "string":
            for count in range(col_spec[1]):
                values[f"{col_type}_{count}"] = ["".join(category.flatten()) for category in np.random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "W", "Z"], size=(col_spec[0], col_spec[0]))]

    df = pd.DataFrame.from_dict(values)

    # 3. Add nulls according to the proportion specified
    for col_type, col_spec in specs.items():
        for col in [col for col in df.columns.values if col_type in col]:
            mask = df[col].sample(frac=col_spec[2]).index
            df.loc[mask, col] = np.nan

    return df

