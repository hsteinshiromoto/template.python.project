import argparse
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import yaml
from typeguard import typechecked

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Get_Filename(ABC):
    def __init__(self, path: Path):
        self.path = path
        super().__init__()

    def list_files(self, pattern: str="*", sort: bool=False):
        files_list = list(filter(Path.is_file, self.path.glob(pattern)))

        output = sorted(files_list, key=os.path.getctime, reverse=True)

        if sort:
            output = sorted(files_list)

        yield from output

    @abstractmethod
    def prep_load(self):
        pass

    @abstractmethod
    def load(self, basename: Union[str, Path]):
        pass

    def load_newest(self, pattern: str="*"):
        newest = self.newest(pattern=pattern)
        return self.load(newest.name)

    def newest(self, pattern: str="*"):
        return self.list_files(pattern=pattern)[0]


class Get_Settings(Get_Filename):
    """Loads settings file

    Args:
        path (Path, optional): Path of seetings file. Defaults to PROJECT_ROOT/"src"/"conf".

    Example:
        >>> settings = Get_Settings().load()
        >>> isinstance(settings, dict)
        True
    """
    def __init__(self, path: Path=PROJECT_ROOT / "src" / "conf"):
        super().__init__(path)

    def load(self, basename: Union[str, Path]="settings.yml"):
        """
        Args:
            basename (str, Path, optional): Basename of settings file. Defaults to "settings.yml".

        Returns:
            dict: settings
        """
        basename = Path(basename)

        with open(str(self.path / basename), 'rb') as file:
            if basename.suffix in [".yml", ".yaml"]:
                return yaml.safe_load(file)

            else:
                return file.open()


@typechecked
def argparse_str2bool(arg: str) -> bool:
    """Convert string to boolean

    Args:
        arg (str): String to be converted to boolean

    Raises:
        argparse.ArgumentTypeError: [description]

    Returns:
        (bool): Boolean value
    """
    if isinstance(arg, bool):
        return arg

    if arg.lower() in ['yes', 'true', 't', 'y', '1', 1]:
        return True

    elif arg.lower() in ['no', 'false', 'f', 'n', '0', 0]:
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
