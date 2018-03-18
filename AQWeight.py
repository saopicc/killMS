import numpy as np
import pylab
from pyrap.tables import table
import Array.ModLinAlg

class AQW():
    def __init__(self,MSName,
                 DataCol="DATA",
                 PredictCol="PREDICT",TBinBox=20):
        self.MSName=MSName
        t=table(self.MSName,ack=False)
        d=t.getcol(DataCol)
        p=t.getcol(PredictCol)
        f=t.getcol("FLAG")
        nr,nch,_=f.shape
        
        t.close()

        d-=p
        d[f]=0.

        self.TBinBox=TBinBox
        self.DATA={}
        self.DATA["data"]=d
        self.DATA["flag"]=f
        self.DATA["A0"]=t.getcol("ANTENNA1")
        self.DATA["A1"]=t.getcol("ANTENNA2")
        self.DATA["W"]=np.zeros_like(t.getcol("IMAGING_WEIGHT"))
        self.na=np.max(self.DATA["A0"])+1
        
    def giveWeigth(self):
        for A0 in range(self.na):
            for A1 in range(self.na):
                self.giveWeigthChunk(A0,A1)
                
    def giveWeigthChunk(self,A0,A1):
        dA0=self.DATA["A0"]
        dA1=self.DATA["A1"]
        f=self.DATA["flag"]
        d=self.DATA["data"]
        C0=(A0==dA0)&(A1==dA1)
        C1=(A1==dA1)&(A0==dA0)
        ind=np.where((A0|A1))[0]
        ds=d[ind]
        fs=f[ind]
        
        nt,nch,_=ds.shape
        nb=self.TBinBox
        for it in range(nb/2,nt-nb/2):
            M=np.zeros((nb,nb),np.complex128)
            i0=it-nb/2
            i1=it+nb/2
            for it0 in range(i0,i1):
                for it1 in range(it0,i1):
                    d0=ds[it0,:,0::3].ravel()
                    d1=ds[it1,:,0::3].ravel()
                    f0=fs[it0,:,0::3].ravel()
                    f1=fs[it1,:,0::3].ravel()
                    df=1-(f0|f1)
                    c=np.sum(d0*d1.conj())/np.count_nonzero(df)
                    M[it0,it1]=c
                    M[it1,it0]=c.conj()

            pylab.clf()
            pylab.imshow(M,interpolation="nearest")
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)

            Minv=Array.ModLinAlg.invSVD(M)
            W=np.sum(Minv,axis=0)
            W=W.reshape((-1,1))*np.ones((1,nch))

            self.DATA["W"][ind,:]=W[:,:]
