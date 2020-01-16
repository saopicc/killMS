from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import optparse
import pickle
import numpy as np
import pylab
from pyrap.tables import table
import killMS.Array.ModLinAlg
from DDFacet.Other import logger
log = logger.getLogger("DiagBL")
from killMS.Data import ClassMS
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
import scipy.misc

class ClassCovMat():
    def __init__(self,**kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        self.DictName="DATA"
        self.TBins=10
        log.print("I am the machine that compute the weights based")
        log.print("  on the (nt x nt)_iAnt:* matrix ")
        
    def giveWeigthParallel(self):
        self.DATA=shared_dict.attach(self.DictName)
        nrow,nch,npol=self.DATA["data"].shape
        self.na=self.DATA["na"]
        ntu=self.DATA["times_unique"].size
        self.DATA["Wa"]=np.zeros((self.na,ntu),np.float32)
        for A0 in range(self.na):
            APP.runJob("giveWeigthAnt:%d"%(A0), 
                       self.giveWeigthAnt,
                       args=(A0,))#,serial=True)
        APP.awaitJobResults("giveWeigthAnt:*", progress="CalcWeight")

    def Finalise(self):
        log.print("Finalising...")
        nrow,nch,npol=self.DATA["data"].shape 
        ntu=self.DATA["times_unique"].size
        nbl=nrow/ntu
        W=self.DATA["W"]#.reshape((ntu,nbl,npol))
        Wa=self.DATA["Wa"].reshape((self.na,ntu,1))
        Wa=Wa*np.ones((1,1,nch))
        dA0=self.DATA["A0"].reshape((ntu,nbl))
        dA1=self.DATA["A1"].reshape((ntu,nbl))
        for A0 in range(self.na):
            for A1 in range(self.na):
                wbl=Wa[A0,:,:]*Wa[A1,:,:]
                C0=((self.DATA["A0"]==A0)&(self.DATA["A1"]==A1))
                C1=((self.DATA["A1"]==A0)&(self.DATA["A0"]==A1))
                ind=np.where(C0|C1)[0]
                W[ind,:]=wbl
                
            
            
    def giveWeigthAnt(self,A0):
        self.DATA=shared_dict.attach(self.DictName)
        self.DATA.reload()
        dA0=self.DATA["A0"]
        dA1=self.DATA["A1"]
        f=self.DATA["flag"]
        d=self.DATA["data"]
        times=self.DATA["times"]
        ntu=self.DATA["times_unique"].size

        indA=np.where((dA0==A0)|(dA1==A0))[0]
        d=d[indA]
        f=f[indA]
        dA0=dA0[indA]
        dA1=dA1[indA]
        nrow,nch,npol=d.shape
        nbl=nrow/ntu
        
        d=d.reshape((ntu,nbl,nch,npol))
        f=f.reshape((ntu,nbl,nch,npol))
        dA0=dA0.reshape((ntu,nbl))
        dA1=dA1.reshape((ntu,nbl))
        
        n=1-f

        DT=3

        Ntu=ntu/DT
        

        
        M=np.zeros((Ntu,Ntu),np.complex128)
        N=np.zeros((Ntu,Ntu),np.float32)
        
        ds0=d[0:DT*Ntu,:,:,:].reshape((Ntu,DT,nbl*nch*npol))
        fs0=n[0:DT*Ntu,:,:,:].reshape((Ntu,DT,nbl*nch*npol))
        ds0=np.sum(ds0,axis=1)
        fs0=np.sum(fs0,axis=1)

        M[:,:]=np.dot(ds0,ds0.T.conj())
        N[:,:]=np.dot(fs0,fs0.T)
        


        # for it0 in range(ntu):
        #     print it0/float(ntu)
        #     ds0=d[it0,:,:,:].reshape((1,nbl*nch*npol))
        #     fs0=n[it0,:,:,:].reshape((1,nbl*nch*npol))
        #     ds1=d.reshape((ntu,nbl*nch*npol))
        #     fs1=n.reshape((ntu,nbl*nch*npol))
        #     M[it0,:]=np.dot(ds0,ds1.T.conj())
        #     N[it0,:]=np.dot(fs0,fs1.T)
            
            
        #     # for it1 in range(it0,ntu):
        #     #     ds1=d[it1,:,:,:].ravel()
        #     #     pp=ds0*ds1.conj()
        #     #     n=np.count_nonzero(pp)
        #     #     spp=np.sum(pp)
        #     #     M[it0,it1]=spp
        #     #     N[it0,it1]=n
        #     #     M[it1,it0]=spp.conj()
        #     #     N[it1,it0]=n
        #print "ok %i"%A0
        # nout=ntu/10
        # M0=scipy.misc.imresize(np.abs(M), (nout,nout))
        # N=scipy.misc.imresize(N, (nout,nout))
        # N[N==0]=1
        # M/=N

        indNZ=(np.sum(N,axis=0)!=0)
        if np.count_nonzero(indNZ)==0:
            return
        Ms=M[indNZ][:,indNZ]
        Ns=N[indNZ][:,indNZ]
        Ms[Ns==0]=np.mean(Ms)
        Ns[Ns==0]=1.
        Ms/=Ns
        
        Minv=killMS.Array.ModLinAlg.invSVD(Ms)
        Wc=np.sum(Minv,axis=0)
        W=np.zeros((indNZ.size,),np.float64)
        W[indNZ]=np.abs(Wc[:])
        W/=np.sum(W)

        for it in range(DT):
            self.DATA["Wa"][A0,it:DT*Ntu:DT]=W[:]


        # pylab.clf()
        # pylab.subplot(1,3,1)
        # pylab.imshow(np.abs(Ms),interpolation="nearest")
        # pylab.subplot(1,3,2)
        # pylab.imshow(np.abs(Ns),interpolation="nearest")
        # pylab.subplot(1,3,3)
        # pylab.imshow(np.abs(Minv),interpolation="nearest")
        # #pylab.subplot(1,3,3)
        # ##pylab.plot(W.ravel())
        # #pylab.plot(self.DATA["W"][ind,0].ravel())
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
