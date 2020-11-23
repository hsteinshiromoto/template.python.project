# -*- coding: utf-8 -*-
import logging
import subprocess
import sys
from pathlib import Path

import click
import dask.dataframe as dd
import pandas as pd
from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
                                stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))
DATA = PROJECT_ROOT / "data"

sys.path.append(PROJECT_ROOT)

from src.base import get_settings
from src.make_logger import make_logger
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


def get_raw_data(basename: str, meta_data: pd.DataFrame, path: Path=DATA / "raw"):

    mask_datetime = meta_data["python_dtypes"] == "datetime64[ns]"
    datetime_columns = list(meta_data[mask_datetime, "python_dtypes"].values)

    dtypes_mapping = {zip(meta_data.loc[mask_datetime, "column_name"].values: 
                        meta_data.loc[mask_datetime, "python_dtypes"].values)}

    if basename.endswith("csv"):
        data = dd.read_csv(str(path / basename), parse_dates=datetime_columns
                            ,date_parser=date_parser, dtypes=dtypes_mapping)

    return data


@click.command()
@click.argument('basename', type=str)
@click.argument('output_filepath', type=click.Path())
def main(basename, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    # Load Settings

    # Load Metadata
    meta_data = pd.read_csv(str(DATA / "meta" / f"{basename}"))

    # Load Raw Data
    raw_data = get_raw_data(basename, meta_data)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
