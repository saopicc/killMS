
import numpy as np
from killMS2.Array import ModLinAlg

def MergeJones(DicoJ0,DicoJ1):
    T0=DicoJ0["t0"][0]
    DicoOut={}
    DicoOut["t0"]=[]
    DicoOut["t1"]=[]
    DicoOut["tm"]=[]
    it=0
    CurrentT0=T0
    
    
    while True:
        DicoOut["t0"].append(CurrentT0)
        T0=DicoOut["t0"][it]
        
        dT0=DicoJ0["t1"]-T0
        dT0=dT0[dT0>0]
        dT1=DicoJ1["t1"]-T0
        dT1=dT1[dT1>0]
        if(dT0.size==0)&(dT1.size==0):
            break
        elif dT0.size==0:
            dT=dT1[0]
        elif dT1.size==0:
            dT=dT0[0]
        else:
            dT=np.min([dT0[0],dT1[0]])
            
        T1=T0+dT
        DicoOut["t1"].append(T1)
        Tm=(T0+T1)/2.
        DicoOut["tm"].append(Tm)
        CurrentT0=T1
        it+=1

        
    DicoOut["t0"]=np.array(DicoOut["t0"])
    DicoOut["t1"]=np.array(DicoOut["t1"])
    DicoOut["tm"]=np.array(DicoOut["tm"])
    
    _,nd,na,nch,_,_=DicoJ0["Jones"].shape
    nt=DicoOut["tm"].size
    DicoOut["Jones"]=np.zeros((nt,nd,na,1,2,2),np.complex64)
    
    nt0=DicoJ0["t0"].size
    nt1=DicoJ1["t0"].size
    
    iG0=np.argmin(np.abs(DicoOut["tm"].reshape((nt,1))-DicoJ0["tm"].reshape((1,nt0))),axis=1)
    iG1=np.argmin(np.abs(DicoOut["tm"].reshape((nt,1))-DicoJ1["tm"].reshape((1,nt1))),axis=1)
    
    
    for itime in range(nt):
        G0=DicoJ0["Jones"][iG0[itime]]
        G1=DicoJ1["Jones"][iG1[itime]]
        DicoOut["Jones"][itime]=ModLinAlg.BatchDot(G0,G1)
        
    
    return DicoOut

