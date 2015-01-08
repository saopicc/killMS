import SharedArray
import ModColor
import numpy as np

def zeros(*args,**kwargs):
    return SharedArray.create(*args,**kwargs)
    
def ToShared(Name,A):

    try:
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)
    except:
        print ModColor.Str("File %s exists, delete it..."%Name)
        DelArray(Name)
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)

    a[:]=A[:]
    return a

def DelArray(Name):
    SharedArray.delete(Name)

def ListNames():
    ll=list(SharedArray.list())
    return [AR.name for AR in ll]
    
def DelAll(key=None):
    ll=ListNames()
    for name in ll:
        if key!=None:
            if key in name: SharedArray.delete(name)
        else:
            SharedArray.delete(name)

def GiveArray(Name):
    try:
        return SharedArray.attach(Name)
    except:
        return None


def DicoToShared(Prefix,Dico):
    DicoOut={}
    for key in Dico.keys():
        if type(Dico[key])!=np.ndarray: continue
        #print "%s.%s"%(Prefix,key)
        ThisKeyPrefix="%s.%s"%(Prefix,key)
        ar=Dico[key].copy()
        Shared=ToShared(ThisKeyPrefix,ar)
        DicoOut[key]=Shared

    return DicoOut

def SharedToDico(Prefix):

    Lnames=ListNames()
    keys=[Name for Name in Lnames if Prefix in Name]
    if len(keys)==0: return False
    DicoOut={}
    for Sharedkey in keys:
        Shared=GiveArray(Sharedkey)
        key=Sharedkey.split(".")[-1]
        DicoOut[key]=Shared


    return DicoOut

