import subprocess
from pathlib import Path
from abc import abstractmethod
from typing import Union
import os

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Get_Filename(object):
    def __init__(self, path: Path):
        self.path = path


    def list_files(self, pattern: str="*", sort: bool=False):
        files_list = list(filter(Path.is_file, self.path.glob(pattern)))

        output = sorted(files_list, key=os.path.getctime, reverse=True)

        if sort:
            output = sorted(files_list)

        yield from output


    def load(self, basename: Union[str, Path]):
        basename = Path(basename)

        with open(str(self.path / basename), 'rb') as file:
            if basename.suffix in [".yml", ".yaml"]:
                return yaml.safe_load(file)

            else:
                return file.open()


    def load_newest(self, pattern: str="*"):
        newest = self.newest(pattern=pattern)
        return self.load(newest.name)


    def newest(self, pattern: str="*"):
        return self.list_files(pattern=pattern)[0]


def get_settings(basename: str="settings.yml"
                ,path: Path=PROJECT_ROOT / "src" / "conf") -> dict:
    """
    Loads settings file

    Args:
        basename (str, optional): Basename of settings file. Defaults to "settings.yml".
        path (Path, optional): Path of seetings file. Defaults to PROJECT_ROOT/"conf".

    Raises:
        exc: Yaml load exception

    Returns:
        dict: settings
    """
    return Get_Filename(path).load(basename)


if __name__ == "__main__":
    pass