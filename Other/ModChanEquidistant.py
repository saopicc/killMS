
import numpy as np

def IsChanEquidistant(ChanFreq):
    df=ChanFreq.flatten()[1::]-ChanFreq.flatten()[0:-1]
    ddf=np.abs(df-np.mean(df))
    ChanEquidistant=int(np.max(ddf)<1.)
    return ChanEquidistant
