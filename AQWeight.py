import numpy as np
import pylab
from pyrap.tables import table
import Array.ModLinAlg
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("AQWeight")

class AQW():
    def __init__(self,MSName,
                 DataCol="DATA",
                 PredictCol="PREDICT",TBinBox=20):
        self.MSName=MSName
        t=table(self.MSName,ack=False)
        print>>log,"Reading data column %s"%DataCol
        d=t.getcol(DataCol)
        
        print>>log,"Reading model column %s"%PredictCol
        p=t.getcol(PredictCol)
        print>>log,"Reading flags %s"%PredictCol
        f=t.getcol("FLAG")
        nr,nch,_=f.shape
        

        d-=p
        d[f]=0.

        self.DATA={}
        self.DATA["data"]=d
        self.DATA["flag"]=f
        print>>log,"Reading other stuff"
        self.DATA["A0"]=t.getcol("ANTENNA1")
        self.DATA["A1"]=t.getcol("ANTENNA2")
        self.DATA["W"]=np.zeros_like(t.getcol("IMAGING_WEIGHT"))
        self.DATA["N"]=np.zeros_like(self.DATA["W"])
        self.na=np.max(self.DATA["A0"])+1
        print>>log,"There are %i antennas"%self.na
        t.close()
        self.TBinBox=TBinBox
        
    def giveWeigth(self):
        for A0 in range(55,self.na):
            for A1 in range(A0+1,self.na):
                self.giveWeigthChunk(A0,A1)
        self.DATA["W"]/=self.DATA["N"]
                
    def giveWeigthChunk(self,A0,A1):
        dA0=self.DATA["A0"]
        dA1=self.DATA["A1"]
        f=self.DATA["flag"]
        d=self.DATA["data"]
        C0=(A0==dA0)&(A1==dA1)
        C1=(A1==dA1)&(A0==dA0)
        ind=np.where((C0|C1))[0]
        C0=(A0==dA0)|(A0==dA1)
        ind=np.where((C0))[0]
        if ind.size==0: return
        ds=d[ind]
        fs=f[ind]
        
        nt,nch,_=ds.shape
        nb=self.TBinBox
        
        
        for it in range(nb/2,nt-nb/2):
            i0=np.max([it-nb/2,0])
            i1=np.min([it+nb/2,nt])
            nbs=i1-i0
            M=np.zeros((nbs,nbs),np.complex128)
            print it
            
            for ic0,it0 in enumerate(range(i0,i1)):
                for ic1,it1 in enumerate(range(i0,i1)):
                    d0=ds[it0,:,0::3].ravel()
                    d1=ds[it1,:,0::3].ravel()
                    f0=fs[it0,:,0::3].ravel()
                    f1=fs[it1,:,0::3].ravel()
                    df=1-(f0|f1)
                    nP=np.count_nonzero(df)
                    if nP==0: continue
                    c=np.sum(d0*d1.conj())/nP
                    
                    M[ic0,ic1]=c
                    M[ic1,ic0]=c.conj()

            Minv=Array.ModLinAlg.invSVD(M)
            W=np.sum(Minv,axis=0)
            W=W.reshape((-1,1))*np.ones((1,nch))

            self.DATA["W"][ind[i0:i1],:]+=np.abs(W[:,:])
            #self.DATA["N"][ind,:][i0:i1]+=1

            pylab.clf()
            pylab.subplot(1,3,1)
            pylab.imshow(np.abs(M),interpolation="nearest")
            pylab.subplot(1,3,2)
            pylab.imshow(np.abs(Minv),interpolation="nearest")
            pylab.subplot(1,3,3)
            #pylab.plot(W.ravel())
            pylab.plot(self.DATA["W"][ind,0].ravel())
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)

            
