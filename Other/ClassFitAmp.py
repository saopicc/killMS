import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassFitAmp")
import killMS.Array.ModLinAlg
from DDFacet.Other import ClassTimeIt
#from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                 denoise_wavelet, estimate_sigma)

MyLogger.setSilent(["ClassFitAmp"])
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
    def __init__(self,gains,nu,Tol=5e-2,Incr=1,RemoveMedianAmp=True):
        self.nt,self.nf,self.na=gains.shape
        self.RemoveMedianAmp=RemoveMedianAmp
        self.G=gains.copy()

        self.G=np.abs(self.G)
        self.GOut=np.zeros_like(self.G)
        self.CentralFreqs=self.nu=nu
        self.NFreq=nu.size
        na=self.na
        self.nbl=(na**2-na)/2
        self.CurrentX=None
        print>>log,"Number of Antennas: %i"%self.na
        print>>log,"Number of Freqs:    %i"%nu.size
        print>>log,"Number of Points:   %i"%(nu.size*self.na**2)
        W=np.array([np.var(self.G[:,:,iAnt]) for iAnt in range(self.na)])
        self.W=W/np.sum(W)
        
        
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
        for iAnt in range(self.na):#[::-1]:
            for iChan in range(self.nf):
                self.GOut[:,iChan,iAnt]=1.
                x=np.arange(self.nt)
                y=self.G[:,iChan,iAnt]
                m0=( np.abs(y[1::]-y[0:-1])>1e-6 )
                m=np.ones_like(y)
                m[1::]=m0[:]
                ind=np.where(m!=0)[0]
                if ind.size<2: continue
                z=np.polyfit(x[ind], y[ind], 10)
                p = np.poly1d(z)
                self.GOut[ind,iChan,iAnt]=p(x[ind])

            if self.RemoveMedianAmp:
                off=np.median(self.G[:,:,iAnt]-self.GOut[:,:,iAnt],axis=1)
                self.GOut[:,:,iAnt]=self.GOut[:,:,iAnt]+off.reshape((-1,1))
            
            
                #self.Plot(iAnt)
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
        
        
        pylab.subplot(2,2,3)
        pylab.imshow(fIm,vmin=vmin,vmax=vmax,aspect="auto",interpolation="nearest")#fIm0)
        
        pylab.subplot(2,2,4)
        #pylab.imshow(Im_n-fIm,vmin=-vmax,vmax=vmax,aspect="auto",interpolation="nearest")#fIm0)
        pylab.imshow(Im_n-fIm,aspect="auto",interpolation="nearest")#fIm0)

        pylab.suptitle(iAnt)
        pylab.draw()
        pylab.show(False)
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
                print w,lStd[iw],lOffCov[iw]

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

        
