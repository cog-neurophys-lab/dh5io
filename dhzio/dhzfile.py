import pathlib
import zarr
import zarr.storage


class DHZFolder:
    """Class for interacting with DAQ-HD (*.dh5) data folders from the Kreiter lab.

    See https://github.com/cog-neurophys-lab/DAQ-HDF5 for the specification of the format for HDF5.
    """

    store: zarr.storage.StoreLike
    root: zarr.Group

    def __init__(self, folder: str | pathlib.Path):
        self.store = zarr.storage.LocalStore(folder)
        self.root = zarr.group(store=self.store, overwrite=False)

    def __del__(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.store.close()
