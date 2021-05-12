import subprocess
from pathlib import Path
from abc import abstractmethod
from typing import Union
import os

import yaml

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))

class Get_Filename(object):
    @abstractmethod
    def __init__(self, path: Path, basename: Union[str, Path]=None
                ,pattern: str="*"):
        self.path = path
        self.basename = Path(basename) if basename else None
        self.pattern = pattern

    @abstractmethod
    def list_files(self, sort_new: bool=False, sort_alphabet: bool=False):
        output = list(filter(Path.is_file, self.path.glob(self.pattern)))

        if sort_new:
            return sorted(output, key=os.path.getctime, reverse=True)

        if sort_alphabet:
            return sorted(output)

        return output

    @abstractmethod
    def load(self):
        with open(str(self.path / self.basename), 'rb') as file:
            if self.basename.suffix in ["yml", "yaml"]:
                return yaml.safe_load(file)

            else:
                return file.open()

    @abstractmethod
    def newest(self):
        return self.list_files(sort_new=True)[0]

    @abstractmethod
    def __iter__(self):
        yield from self.list_files()

    @abstractmethod
    def __next__(self): # Python 2: def next(self)
        return self



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
    with open(str(path / basename), 'r') as stream:
        try:
            settings = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            raise exc

    return settings


if __name__ == "__main__":
    get_file = Get_File(path=Path(__file__).resolve().parent)

    for item in get_file:
        print(item)

    print(f"Newest: {get_file.newest()}\n")

    print(f"Sort by date: {get_file.list_files(sort_new=True)}\n")

    print(f"Sort alphabetically: {get_file.list_files(sort_alphabet=True)}\n")

    # for item in filter(Path.is_file, Path(__file__).resolve().parent.glob("*")):
    #     print(item)
