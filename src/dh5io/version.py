import tomllib
import os


def get_version():
    here = os.path.dirname(__file__)
    pyproject_path = os.path.join(here, "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]
