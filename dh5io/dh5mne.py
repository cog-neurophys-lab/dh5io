import mne
import pathlib
import dh5file
import numpy as np

# See 
# - https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html#tut-creating-data-structures
# - https://mne.tools/stable/auto_tutorials/raw/20_event_arrays.html#sphx-glr-auto-tutorials-raw-20-event-arrays-py
# - https://mne.tools/stable/generated/mne.io.RawArray.html
# - https://mne.tools/stable/generated/mne.EpochsArray.html#mne.EpochsArray
# - https://mne.tools/stable/generated/mne.EvokedArray.html#mne.EvokedArray
# - https://mne.tools/stable/generated/mne.create_info.html

def read_cont_to_mne_raw(filename : str | pathlib.Path, contIds : list[int] | None=None) -> mne.io.Raw:
    """Read a DAQ-HDF5 file into a MNE Raw object.

    Parameters
    ----------
    filename : str | pathlib.Path
        The path to the DAQ-HDF5 file.
    preload : bool
        If True, all data are loaded at initialization. If False (default),
        data are not read 

    Returns
    -------
    raw : mne.io.Raw
        The MNE Raw object.
    """

    dh5 = dh5file.DH5File(filename)
    return cont_to_mne_raw(dh5, contIds=contIds)

def cont_to_mne_raw(dh5 : dh5file.DH5File, contIds:list[int]=None) -> mne.io.Raw:
    """Read a DAQ-HDF5 file into a MNE Raw object.

    Parameters
    ----------
    dh5 : dh5file.DH5File
        The DAQ-HDF5 file.
    preload : bool
        If True, all data are loaded at initialization. If False (default),
        data are not read 

    Returns
    -------
    raw : mne.io.Raw
        The MNE Raw object.
    """

    if contIds is None:
        contIds = dh5.get_cont_group_ids()

    allCont = []
    for id in contIds:
        allCont.append(dh5.get_cont_group_by_id(id))
    

    nChannels = len(contIds)
    nSamples = 1024
    info = mne.create_info(ch_names=nChannels, sfreq=1000, ch_types='misc')
    data = np.zeros(nChannels, nSamples)
    return mne.io.RawArray(data=data, info=info)