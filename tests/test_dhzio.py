import dhzio
import dhzio.cont
import zarr
import numpy as np

import dhzio.dhzfile


def test_dhzio(tmp_path):
    folder = tmp_path / "test.dhz"

    dhzfile = dhzio.dhzfile.DHZFolder(folder)

    # create a new CONT group
    cont_group = dhzio.cont.create_empty_cont_group(
        dhzfile.root,
        cont_group_id=10,
        nSamples=1000,
        nChannels=32,
        sample_period_ns=1000,
    )
