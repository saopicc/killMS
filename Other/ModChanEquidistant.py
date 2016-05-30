
import numpy as np

def IsChanEquidistant(ChanFreq):

    if ChanFreq.size>1:
        df=ChanFreq.flatten()[1::]-ChanFreq.flatten()[0:-1]
        ddf=np.abs(df-np.mean(df))
        ChanEquidistant=int(np.max(ddf)<1.)
    else:
        ChanEquidistant=0
    return ChanEquidistant
