import pathlib
import numpy
import h5py


class DH5File:
    """Class for interacting with DAQ-HDF5 (*.dh5) files from the Kreiter lab. 
    
    The file format ist based on HDF5. See https://github.com/cog-neurophys-lab/DAQ-HDF5 for
    the specification of the format.
    """

    file: h5py.File

    def __init__(self, filename: str | pathlib.Path, mode="r"):
        self.file = h5py.File(filename, mode)

    def __str__(self):
        return f"""DAQ-HDF5 File (version {self.get_version()}) {self.file.filename:s} containing:
          ├─── {len(self.get_cont_groups()):5d} CONT Groups: {self.get_cont_group_names()}
          ├─── {len(self.get_spike_groups()):5d} SPIKE Groups: {self.get_spike_group_names()}
          ├─── {len(self.get_events()):5d} Events
          └─── {len(self.get_trialmap()):5d} Trials in TRIALMAP

        """
    
    def get_version(self) -> int | None:
        return self.file.attrs.get("FILEVERSION")

    # cont groups
    def get_cont_groups(self) -> list[h5py.Group]:
        return [self.file[name] for name in self.get_cont_group_names()]

    def get_cont_group_names(self) -> list[str]:
        return [
            name
            for name in self.file.keys()
            if name.startswith("CONT") and isinstance(self.file[name], h5py.Group)
        ]
    
    def get_cont_group_ids(self) -> list[int]:
        return [
            int(name.split("CONT")[1])
            for name in self.file.keys()
            if name.startswith("CONT") and isinstance(self.file[name], h5py.Group)
        ]

    def get_cont_group_by_id(self, id: int) -> h5py.Group:
        contGroup = self.file.get(f"CONT{id}")
        if contGroup is None:
            raise DH5Error(f"CONT{id} does not exist in {self.file.filename}")
        return contGroup
    
    def get_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        return numpy.array(self.get_cont_group_by_id(cont_id).get("DATA")[:])
    
    def get_calibrated_cont_data_by_id(self, cont_id: int) -> numpy.ndarray:
        calibration = self.get_cont_group_by_id(cont_id).attrs.get("Calibration")
        if calibration is None: return self.get_cont_data_by_id(cont_id)
        return self.get_cont_data_by_id(cont_id) * calibration

    def get_cont_size(self, cont_id) -> tuple[int, int]:
        nSamples, nChannels = self.get_cont_data_by_id(cont_id).shape
        return (nSamples, nChannels)



    # spike groups
    def get_spike_groups(self) -> list[h5py.Group]:
        return [self.file[name] for name in self.get_spike_group_names()]

    def get_spike_group_names(self) -> list[str]:
        return [
            name
            for name in self.file.keys()
            if name.startswith("SPIKE") and isinstance(self.file[name], h5py.Group)
        ]

    def get_spike_group_by_id(self, id: int) -> h5py.Group | None:
        return self.file.get(f"SPIKE{id}")

    def get_cont_index_by_id(self, cont_id: int) -> h5py.Dataset:
        return self.get_cont_group_by_id(cont_id).get("INDEX")


    def get_trialmap(self) -> h5py.Dataset | None:
        return self.file.get("TRIALMAP")
    
    def get_events(self) -> h5py.Dataset | None:
        return self.file.get("EV02")

    def __del__(self):
        self.file.close()

    @staticmethod
    def get_cont_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("CONT"))

    @staticmethod
    def get_spike_id_from_name(name: str) -> int | None:
        return int(name.lstrip("/").lstrip("SPIKE"))
    

class DH5Error(Exception):
    pass