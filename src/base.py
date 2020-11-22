import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], 
stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))


def get_settings(basename: str="settings.yml", path: Path=PROJECT_ROOT / "conf") -> dict:
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
