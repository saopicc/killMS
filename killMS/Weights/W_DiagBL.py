from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import optparse
import pickle
import numpy as np
from DDFacet.Other import logger
log = logger.getLogger("DiagBL")
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError

class ClassCovMat():
    def __init__(self,**kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        self.DictName="DATA"
        log.print("I am the machine that compute the weights based")
        log.print("  on the diag(nt x nt)_iAnt:jAnt matrix")

    def giveWeigthParallel(self):
        self.DATA=shared_dict.attach(self.DictName)
        self.na=self.DATA["na"]
        for A0 in range(0,self.na):
            for A1 in range(A0+1,self.na):
                    APP.runJob("giveWeigthChunk:%d:%d"%(A0,A1), 
                               self.giveWeigthChunk,
                               args=(A0,A1,self.DATA.readwrite()))#,serial=True)
        APP.awaitJobResults("giveWeigthChunk:*", progress="CalcWeight")
        
    def Finalise(self):
        ind=(self.DATA["N"]==0)
        self.DATA["N"][ind]=1
        self.DATA["W"]/=self.DATA["N"]

    def giveWeigth(self):
        for A0 in range(55,self.na):
            for A1 in range(A0+1,self.na):
                self.giveWeigthChunk(A0,A1)

    def giveWeigthChunk(self,A0,A1,DATA):
        self.DATA=shared_dict.attach(self.DictName)
        self.DATA.reload()
        dA0=self.DATA["A0"]
        dA1=self.DATA["A1"]
        f=self.DATA["flag"]
        d=self.DATA["data"]
        C0=(A0==dA0)&(A1==dA1)
        C1=(A1==dA1)&(A0==dA0)
        ind=np.where((C0|C1))[0]
        # C0=(A0==dA0)|(A0==dA1)
        # ind=np.where((C0))[0]
        if ind.size==0: return
        ds=d[ind]
        fs=f[ind]
        
        nt,nch,_=ds.shape
        #nb=100
        nb=self.TBinBox
        
        for it in range(nb/2,nt-nb/2):
            #print it,"/",nt
            i0=np.max([it-nb/2,0])
            i1=np.min([it+nb/2,nt])
            nbs=i1-i0
            M=np.zeros((nbs,nbs),np.complex128)
            for ic0,it0 in enumerate(range(i0,i1)):
                d0=ds[it0,:,0::3].ravel()
                f0=fs[it0,:,0::3].ravel()
                df=1-f0
                nP=np.count_nonzero(df)
                if nP==0: continue
                c=np.sum(d0*d0.conj())/nP
                
                M[ic0,ic0]=c

                # for ic1,it1 in enumerate(range(i0,i1)):
                #     d0=ds[it0,:,0::3].ravel()
                #     d1=ds[it1,:,0::3].ravel()
                #     f0=fs[it0,:,0::3].ravel()
                #     f1=fs[it1,:,0::3].ravel()
                #     df=1-(f0|f1)
                #     nP=np.count_nonzero(df)
                #     if nP==0: continue
                #     c=np.sum(d0*d1.conj())/nP
                    
                #     M[ic0,ic1]=c
                #     M[ic1,ic0]=c.conj()

            indNZ=(np.sum(M,axis=0)!=0)



            if np.count_nonzero(indNZ)==0: continue
            M=np.diag(np.diag(M)+1.)
            # # #####################
            # M=np.sum(np.abs(ds[:,:])**2,axis=1)
            # M=np.diag(M)
            # indNZ=(np.sum(M,axis=0)!=0)
            # #####################
            Minv=Array.ModLinAlg.invSVD(M[indNZ][:,indNZ])
            Wc=np.sum(Minv,axis=0)
            Wc=Wc.reshape((-1,1))*np.ones((1,nch))
            W=np.zeros((indNZ.size,nch),np.float64)
            W[indNZ,:]=np.abs(Wc[:,:])
            #####################
            # W[indNZ,:]/=np.sum(W[indNZ,:])
            #####################
            O=np.zeros_like(W)
            O[indNZ,:]=1
            
            self.DATA["W"][ind[i0:i1],:]+=W[:,:]
            self.DATA["N"][ind[i0:i1],:]+=O[:,:]
            # mm=np.max(self.DATA["W"])
            # if mm>1e5:
                
            
            #     pylab.clf()
            #     pylab.subplot(1,3,1)
            #     pylab.imshow(np.abs(M),interpolation="nearest")
            #     pylab.subplot(1,3,2)
            #     pylab.imshow(np.abs(Minv),interpolation="nearest")
            #     pylab.subplot(1,3,3)
            #     #pylab.plot(W.ravel())
            #     pylab.plot(self.DATA["W"][ind,0].ravel())
            #     pylab.draw()
            #     pylab.show(False)
            #     pylab.pause(0.1)
            #     stop
