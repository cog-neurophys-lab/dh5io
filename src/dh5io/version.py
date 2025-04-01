import importlib.metadata


def get_version():
    return importlib.metadata.version("dh-format")
