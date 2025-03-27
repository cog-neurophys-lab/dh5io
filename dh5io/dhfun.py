"""
An API similar to the MATLAB API for the DAQ-HDF5 file format.
"""
# %  Function reference:
# %
# %  ----- General information, debugging ------------
# %
# %  DH.GETVERSION
# %  DH.LISTOPENFIDS
# %
# %  --- General file service ------------------------
# %
# %  DH.OPEN
# %  DH.CLOSE
# %  DH.GETFIDINFO
# %  DH.GETOPERATIONINFOS
# %  DH.GETDAQVERSION (-)
# %
# %  --- DAQ-HDF V1 continuous recordings ------------
# %
# %  DH.CREATECR (-)
# %  DH.READCR
# %  DH.WRITECR
# %  DH.GETCRSIZE
# %  DH.GETCRADCBITWIDTH
# %  DH.GETCRSAMPLEPERIOD
# %  DH.GETCRSTARTTIME
# %  DH.GETCRMAXVOLTAGERANGE
# %  DH.GETCRMINVOLTAGERANGE
# %  DH.GETCRCALINFO
# %  DH.SETCRCALINFO (-)
# %
# %  --- DAQ-HDF V1 event triggers -------------------
# %
# %  DH.CREATEEV (-)
# %  DH.READEV
# %  DH.WRITEEV
# %  DH.GETEVSIZE
# %
# %  --- DAQ-HDF all versions TD01 records -----------
# %
# %  DH.CREATETD
# %  DH.READTD
# %  DH.WRITETD
# %  DH.GETTDSIZE
# %
# %  --- DAQ-HDF V2 CONT nTrodes ---------------------
# %
# %  DH.CREATECONT
# %  DH.ENUMCONT
# %  DH.READCONT
# %  DH.WRITECONT
# %  DH.READCONTINDEX
# %  DH.WRITECONTINDEX
# %  DH.GETCONTSIZE
# %  DH.GETCONTINDEXSIZE
# %  DH.GETCONTSAMPLEPERIOD
# %  DH.SETCONTSAMPLEPERIOD
# %  DH.GETCONTCALINFO
# %  DH.SETCONTCALINFO
# %  DH.GETCONTCHANDESC
# %  DH.SETCONTCHANDESC (-)
# %
# %  --- DAQ-HDF V2 SPIKE nTrodes --------------------
# %
# %  DH.CREATESPIKE
# %  DH.ENUMSPIKE
# %  DH.READSPIKE
# %  DH.WRITESPIKE
# %  DH.READSPIKEINDEX
# %  DH.WRITESPIKEINDEX
# %  DH.ISCLUSTERINFO_PRESENT
# %  DH.READSPIKECLUSTER
# %  DH.WRITESPIKECLUSTER
# %  DH.GETSPIKESIZE
# %  DH.GETNUMBERSPIKES
# %  DH.GETSPIKESAMPLEPERIOD
# %  DH.GETSPIKEPARAMS
# %  DH.GETSPIKECHANDESC (-)
# %  DH.SETSPIKECHANDESC (-)
# %
# %  --- WAVELET interface ---------------------------
# %  DH.CREATEWAVELET
# %  DH.ENUMWAVELET
# %  DH.READWAVELET
# %  DH.WRITEWAVELET
# %  DH.READWAVELETINDEX
# %  DH.WRITEWAVELETINDEX
# %  DH.GETWAVELETSIZE
# %  DH.GETWAVELETINDEXSIZE
# %  DH.GETWAVELETSAMPLEPERIOD
# %  DH.SETWAVELETSAMPLEPERIOD
# %  DH.GETWAVELETCHANDESC        (-)
# %  DH.SETWAVELETCHANDESC        (-)
# %  DH.GETWAVELETFAXIS
# %  DH.SETWAVELETFAXIS
# %  DH.GETWAVELETMORLETPARAMS
# %  DH.SETWAVELETMORLETPARAMS
# %
# %  --- DAQ-HDF V2 EV02 triggers --------------------
# %
# %  DH.CREATEEV2
# %  DH.READEV2
# %  DH.WRITEEV2
# %  DH.GETEV2SIZE
# %
# %  ---------- TRIALMAP interface -------------------
# %
# %  DH.GETTRIALMAP
# %  DH.SETTRIALMAP
# %
# %  ---------- MARKER interface ---------------------
# %
# %  DH.ENUMMARKERS
# %  DH.GETMARKER
# %  DH.SETMARKER
# %
# %  ---------- INTERVAL interface -------------------
# %
# %  DH.ENUMINTERVALS
# %  DH.GETINTERVAL
# %  DH.SETINTERVAL
# %
# %
# %  -------------------------------------------------
# %
# %  Functions marked with (-) are not implemented.
# %  They will be implemented on demand.

raise ImportError("Nothing here yet")
