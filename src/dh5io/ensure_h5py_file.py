import h5py
import pathlib


def ensure_h5py_file(func, mode="r"):
    def wrapper(file, *args, **kwargs):
        if isinstance(file, (str, pathlib.Path)):
            with h5py.File(file, mode=mode) as f:
                return func(f, *args, **kwargs)
        elif isinstance(file, h5py.File):
            return func(file, *args, **kwargs)
        else:
            raise TypeError("file must be a h5py.File or a str or pathlib.Path")

    return wrapper
