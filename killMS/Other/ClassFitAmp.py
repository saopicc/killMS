from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("ClassFitAmp")
import killMS.Array.ModLinAlg
from DDFacet.Other import ClassTimeIt
#from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                 denoise_wavelet, estimate_sigma)

logger.setSilent(["ClassFitAmp"])
#from tvd import TotalVariationDenoising

def Dot(*args):
    P=1.
    for M in args:
        #P=np.dot(np.complex128(P),np.complex128(M))
        P=np.dot(P,M)
    return P

# iDir=14; S=np.load("L229509_merged.npz"); G=S["Sols"]["G"][:,:,:,iDir,0,0]; f=S["FreqDomains"].mean(axis=1)

def Norm(G,iRef=0):
    nf,na=G.shape
    for iFreq in range(nf):
        g0=G[iFreq,iRef]
        G[iFreq]*=g0.conj()/np.abs(g0)
    

        
def test(G,f):
    # nf,na=G.shape
    # #na=3
    # t=np.random.randn(na)*0.01
    # c=np.random.randn(na)*np.pi/10
    # G=TECToZ(t.reshape((1,-1)),c.reshape((1,-1)),f.reshape((-1,1)))

    AmpMachine=ClassFitAmp(G,f)
    AmpMachine.doSmooth()


class ClassFitAmp():
    def __init__(self,gains,nu,Tol=5e-2,Incr=1,RemoveMedianAmp=True,LogMode=False):
        self.nt,self.nf,self.na=gains.shape
        self.RemoveMedianAmp=RemoveMedianAmp
        self.G=gains.copy()
        self.LogMode=LogMode
        self.G=np.abs(self.G)

        if self.LogMode:
            self.G[self.G==0]=1e-6
            self.G=np.log(self.G)
            
        self.GOut=np.zeros_like(self.G)
        self.CentralFreqs=self.nu=nu
        self.NFreq=nu.size
        na=self.na
        self.nbl=(na**2-na)//2
        self.CurrentX=None
        log.print("Number of Antennas: %i"%self.na)
        log.print("Number of Freqs:    %i"%nu.size)
        log.print("Number of Points:   %i"%(nu.size*self.na**2))
        # W=np.array([np.var(self.G[:,:,iAnt]) for iAnt in range(self.na)])
        # if np.sum(W)==0: stop
        # self.W=W/np.sum(W)

        
    # def doSmooth(self):
    #     for iTime in range(self.nt):
    #         for iFreq in range(self.nf):
    #             self.doSmoothThisTF(iTime,iFreq)
    #     return self.GOut
    # def doSmoothThisTF(self,iTime,iFreq):
    #     g=self.G[iTime,iFreq].ravel()
    #     Y=( g.reshape((-1,1)) * g.conj().reshape((1,-1)) )
    #     for iAnt in range(self.na):
    #         self.GOut[iTime,iFreq,iAnt]=np.sqrt(np.sum(self.W*Y[iAnt]))

    def doSmooth(self):
        vmax=2.
        vmin=.3
        for iAnt in range(self.na):#[::-1]:
            for iChan in range(self.nf):
                # print
                # print "ichan %i %i (%i,%i)"%(iChan,iAnt,self.nf,self.na)
                mout=np.zeros((self.nt,),np.float32)
                ii=0
                while True:
                    self.GOut[:,iChan,iAnt]=1.
                    x=np.arange(self.nt)
                    y=self.G[:,iChan,iAnt].copy()
                    m0=( np.abs(y[1::]-y[0:-1])>1e-6 )
                    m=np.ones_like(y)
                    m[1::]=m0[:]
                    ind=np.where((m!=0) & (mout==0))[0]
                    # print np.count_nonzero(ind)
                    yc=np.ones_like(y)
                    if ind.size<2:
                        break
                    yc[ind]=y[ind]
                    xc=x[ind]
                    z=np.polyfit(x[ind], y[ind], 10)
                    p = np.poly1d(z)
                    px=np.abs(p(x))
                    pp=px/np.mean(px)*np.median(y[ind])
                    self.GOut[:,iChan,iAnt]=pp

                    # import pylab
                    # pylab.clf()
                    # pylab.plot()
                    
                    # ##########################
                    resid=y-pp
                    rabs=np.abs(resid-np.median(np.abs(resid)))
                    sig=1.48*np.median(rabs)
                    s_mout0=np.count_nonzero(mout)
                    mout[rabs>5.*sig]=1.
                    s_mout1=np.count_nonzero(mout)
                    if s_mout0==s_mout1:
                        self.GOut[:,iChan,iAnt]=np.abs(p(x))
                        # print iChan,ii,s_mout0
                        break
                    ii+=1
                    ## print ii
                Ag=np.abs(self.GOut[:,iChan,iAnt])
                #if Ag.max()/Ag.min()>2: self.GOut[:,iChan,iAnt].fill(np.median(self.G[:,:,iAnt]))
                if ind.size<2:
                    continue
                    
            if self.RemoveMedianAmp:
                off=np.median(self.G[:,:,iAnt]-self.GOut[:,:,iAnt],axis=1)
                self.GOut[:,:,iAnt]=self.GOut[:,:,iAnt]+off.reshape((-1,1))
            if not self.LogMode:
                self.GOut[self.GOut>vmax]=vmax
                self.GOut[self.GOut<vmin]=vmin
            
            # self.Plot(iAnt)

        if self.LogMode:
            self.GOut=np.exp(self.GOut)
        return self.GOut

    def indUnique(self,a):
        unq, unq_idx, unq_cnt = np.unique(a, return_inverse=True, return_counts=True)
        return unq_cnt==1



        
    def Plot(self,iAnt):
        Im_n=self.G[:,:,iAnt]
        fIm=self.GOut[:,:,iAnt]
        import pylab
        vmin,vmax=Im_n.min(),Im_n.max()
        import pylab
        pylab.gray()
        pylab.clf()
        pylab.subplot(2,2,1)
        #pylab.imshow(Im,vmin=vmin,vmax=vmax)
        
        pylab.subplot(2,2,2)
        pylab.imshow(Im_n,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest")
        pylab.colorbar()
        pylab.title("In Jones")
        
        pylab.subplot(2,2,3)
        pylab.imshow(fIm,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest")#fIm0)
        pylab.colorbar()
        pylab.title("FitJones")
        
        pylab.subplot(2,2,4)
        #pylab.imshow(Im_n-fIm,vmin=-vmax,vmax=vmax,aspect="auto",interpolation="nearest")#fIm0)
        pylab.imshow(Im_n-fIm,aspect="auto",interpolation="nearest")#fIm0)
        pylab.colorbar()
        pylab.title("resid")

        pylab.suptitle(iAnt)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        
        
    
    def doSmoothDeNoise(self):

        Nw=1
        for iAnt in range(self.na)[::-1]:
            Im_n=self.G[:,:,iAnt]
            wtry=np.linspace(0.001,1,Nw)
            lIm=[]
            lStd=np.zeros((Nw,),np.float32)
            lOffCov=np.zeros((Nw,),np.float32)
            sigma_est = estimate_sigma(Im_n, average_sigmas=True)
            for iw in range(Nw):
                w=wtry[iw]
                #fIm=denoise_tv_chambolle(Im_n, weight=w)
                M=Im_n.max()
                fIm=denoise_bilateral(Im_n/M, sigma_range=0.1, sigma_spatial=15, multichannel=False)
                fIm*=M
                # subject = TotalVariationDenoising(Im_n)
                # fIm = subject.generate()


                lIm.append(fIm)

                lStd[iw]=np.std(fIm)
                ###################################"
                OffCov=(np.sum(fIm[1:,:]*fIm[:-1,:])+np.sum(fIm[:,1:]*fIm[:,:-1]))/fIm.size
                lOffCov[iw]=OffCov
                Tot=lOffCov[iw]+lStd[iw]
                print(w,lStd[iw],lOffCov[iw])

                import pylab
                vmin,vmax=Im_n.min(),Im_n.max()
                import pylab
                pylab.gray()
                pylab.clf()
                pylab.subplot(2,2,1)
                #pylab.imshow(Im,vmin=vmin,vmax=vmax)
                
                pylab.subplot(2,2,2)
                pylab.imshow(Im_n,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest")

            
                pylab.subplot(2,2,3)
                pylab.imshow(fIm,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest")#fIm0)
                
                pylab.subplot(2,2,4)
                pylab.imshow(Im_n-fIm,vmin=-vmax,vmax=vmax,aspect="auto",interpolation="nearest")#fIm0)
                
                pylab.title("w=%f"%w)
                pylab.draw()
                pylab.show(False)
                pylab.pause(0.1)

            # import pylab
            # pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.plot(lStd)
            # pylab.subplot(1,2,2)
            # pylab.plot(lOffCov)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            
            ii=np.argmin(np.abs(sigma_est-np.array(lStd)))    
            self.GOut[:,:,iAnt]=lIm[ii]


            
        return self.GOut

        
