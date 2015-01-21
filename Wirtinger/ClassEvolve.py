

import numpy as np
import pylab
from Array import NpShared

class ClassModelEvolution():
    def __init__(self,iAnt,WeightType="exp",WeigthScale=1,order=1,StepStart=10,BufferNPoints=10,sigQ=0.01,DoEvolve=True):
        self.WeightType=WeightType 
        self.WeigthScale=WeigthScale*60. #in min
        self.order=order
        self.StepStart=StepStart
        self.iAnt=iAnt
        self.BufferNPoints=BufferNPoints
        self.sigQ=sigQ
        self.DoEvolve=DoEvolve
    

    def Evolve0(self,Gin,Pa):
        done=NpShared.GiveArray("SolsArray_done")
        indDone=np.where(done==1)[0]
        Q=NpShared.GiveArray("SharedCovariance_Q")[self.iAnt]
        #print indDone.size
        #print "mean",np.mean(Q)
        if indDone.size==0: return Pa+Q
        t0=NpShared.GiveArray("SolsArray_t0")[indDone]
        t1=NpShared.GiveArray("SolsArray_t1")[indDone]
        tm=NpShared.GiveArray("SolsArray_tm")[indDone]


        G=NpShared.GiveArray("SolsArray_G")[indDone][:,self.iAnt,:,:,:]

        
        nt,nd,npol,_=G.shape

        #if nt<=self.StepStart: return None

        if nt>self.BufferNPoints:
            G=G[-self.BufferNPoints::,:,:,:]

        G=G.copy()

        nt,_,_,_=G.shape
        NPars=nd*npol*npol
        G=G.reshape((nt,NPars))


        F=np.ones((NPars,),np.complex128)

        for iPar in range(NPars):
            #g_t=G[:,iPar][-1]
            #ThisG=Gin.ravel()[iPar]
            #ratio=1.+(ThisG-g_t)/g_t
            g_t=G[:,iPar]
            ThisG=Gin.ravel()[iPar]
            #ratio=1.+np.std(g_t)
            ratio=(ThisG-np.mean(g_t))
            F[iPar]=ratio#/np.sqrt(2.)


        
        PaOut=np.zeros_like(Pa)
        # Q=np.diag(np.ones((PaOut.shape[0],)))*(self.sigQ**2)


        PaOut=F.reshape((NPars,1))*Pa*F.reshape((1,NPars)).conj()+Q
        # print F
        # print Q
        # stop
        return PaOut
        


    def Evolve(self,x,P,CurrentTime):
        done=NpShared.GiveArray("SolsArray_done")
        indDone=np.where(done==1)[0]

        t0=NpShared.GiveArray("SolsArray_t0")[indDone]
        t1=NpShared.GiveArray("SolsArray_t1")[indDone]
        tm=NpShared.GiveArray("SolsArray_tm")[indDone]


        G=NpShared.GiveArray("SolsArray_G")[indDone][:,self.iAnt,:,:,:]

        
        nt,nd,npol,_=G.shape

        if nt<=self.StepStart: return None,None

        if nt>self.BufferNPoints:
            G=G[-nt::,:,:,:]
            tm=tm[-nt::]

        G=G.copy()
        tm0=tm.copy()
        tm0=tm-tm[-1]
        ThisTime=CurrentTime-tm[-1]

        nt,_,_,_=G.shape
        NPars=nd*npol*npol
        G=G.reshape((nt,NPars))


        Gout=np.zeros((nd*npol*npol),dtype=G.dtype)
        Gout[:]=G[-1]


        F=np.ones((NPars,),np.complex128)
        if self.DoEvolve:
            if self.WeightType=="exp":
                w=np.exp(-tm0/self.WeigthScale)
                w/=np.sum(w)
                w=w[::-1]
            dx=1e-6
            for iPar in range(NPars):
                g_t=G[:,iPar]
                g_r=g_t.real
                g_i=g_t.imag
                
                ####
                z_r0 = np.polyfit(tm0, g_r, self.order, w=w)
                z_i0 = np.polyfit(tm0, g_i, self.order, w=w)
                poly_r = np.poly1d(z_r0)
                poly_i = np.poly1d(z_i0)
                x0_r=poly_r(ThisTime)
                x0_i=poly_i(ThisTime)
                Gout[iPar]=x0_r+1j*x0_i
                
                ####
                z_r1 = np.polyfit(tm0, g_r+dx, self.order, w=w)
                z_i1 = np.polyfit(tm0, g_i+dx, self.order, w=w)
                poly_r = np.poly1d(z_r1)
                poly_i = np.poly1d(z_i1)
                x1_r=poly_r(ThisTime)
                x1_i=poly_i(ThisTime)
                
                dz=((x0_r-x1_r)+1j*(x0_i-x1_i))/dx
                F[iPar]=dz/np.sqrt(2.)

            # if self.iAnt==0:
            #     xx=np.linspace(tm0.min(),tm0.max(),100)
            #     pylab.clf()
            #     pylab.plot(tm0, g_r)
            #     pylab.plot(xx, poly_r(xx))
            #     pylab.scatter([ThisTime],[x1_r])
            #     pylab.draw()
            #     pylab.show(False)
            #     pylab.pause(0.1)

            
        # if self.iAnt==0:
        #     pylab.clf()
        #     pylab.imshow(np.diag(F).real,interpolation="nearest")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)


        Pa=P[self.iAnt]
        PaOut=np.zeros_like(Pa)
        Q=np.diag(np.ones((PaOut.shape[0],)))*(self.sigQ**2)
        PaOut=F.reshape((NPars,1))*Pa*F.reshape((1,NPars)).conj()+Q
        
        

        Gout=Gout.reshape((nd,npol,npol))


        return Gout,PaOut
  
