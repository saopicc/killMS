#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from killMS.Array import NpShared

class ClassModelEvolution():
    def __init__(self,iAnt,iChanSol,WeightType="exp",WeigthScale=1,order=1,StepStart=5,BufferNPoints=10,sigQ=0.01,DoEvolve=True,IdSharedMem=""):
        self.WeightType=WeightType 
        self.WeigthScale=WeigthScale*60. #in min
        self.order=order
        self.StepStart=StepStart
        self.iAnt=iAnt
        self.iChanSol=iChanSol
        self.BufferNPoints=BufferNPoints
        self.sigQ=sigQ
        self.DoEvolve=DoEvolve
        self.IdSharedMem=IdSharedMem

    def Evolve0(self,Gin,Pa,kapa=1.):
        done=NpShared.GiveArray("%sSolsArray_done"%self.IdSharedMem)
        
        
        indDone=np.where(done==1)[0]
        #print kapa
        #print type(NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem))
        Q=kapa*NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem)[self.iChanSol,self.iAnt]
        #print indDone.size
        #print "mean",np.mean(Q)


        ##########
        # Ptot=Pa+Q
        # #nt,_,_,_=Gin.shape
        # #print Gin.shape
        # g=Gin
        # gg=g.ravel()
        # #gg+=(np.random.randn(*gg.shape)+1j*np.random.randn(*gg.shape))*np.sqrt(np.diag(Ptot))/np.sqrt(2.)

        # # print Pa.shape,Q.shape
        # # print np.diag(Pa)
        # # print np.diag(Q)
        # # print np.diag(Ptot)
        # # print

        # return Ptot
        ##############

        # return Pa+Q
        if indDone.size<2: return Pa+Q
        
        # #########################
        # take scans where no solve has been done into account
        G=NpShared.GiveArray("%sSolsArray_G"%self.IdSharedMem)[indDone][:,self.iChanSol,self.iAnt,:,0,0]
        Gm=np.mean(G,axis=-1)
        dG=Gm[:-1]-Gm[1:]
        done0=NpShared.GiveArray("%sSolsArray_done"%self.IdSharedMem)
        done1=np.zeros((done0.size,),int)
        done1[1:1+dG.size]=(dG!=0)
        done1[0]=1
        done=(done0&done1)
        indDone=np.where(done==1)[0]
        if indDone.size<2: return Pa+Q
        # #########################
        
        t0=NpShared.GiveArray("%sSolsArray_t0"%self.IdSharedMem)[indDone]
        t1=NpShared.GiveArray("%sSolsArray_t1"%self.IdSharedMem)[indDone]
        tm=NpShared.GiveArray("%sSolsArray_tm"%self.IdSharedMem)[indDone]


        G=NpShared.GiveArray("%sSolsArray_G"%self.IdSharedMem)[indDone][:,self.iChanSol,self.iAnt,:,:,:]

        
        nt,nd,npolx,npoly=G.shape

        #if nt<=self.StepStart: return None

        if nt>self.BufferNPoints:
            G=G[-self.BufferNPoints::,:,:,:]
            tm=tm[-self.BufferNPoints::]

        G=G.copy()

        nt,_,_,_=G.shape
        NPars=nd*npolx*npoly
        G=G.reshape((nt,NPars))

        F=np.ones((NPars,),G.dtype)
        PaOut=np.zeros_like(Pa)

        tm0=tm.copy()
        tm0=np.abs(tm-tm[-1])
        w=np.exp(-tm0/self.WeigthScale)
        w/=np.sum(w)
        w=w[::-1]

        for iPar in range(NPars):
            #g_t=G[:,iPar][-1]
            #ThisG=Gin.ravel()[iPar]
            #ratio=1.+(ThisG-g_t)/g_t
            g_t=G[:,iPar]
            ThisG=Gin.ravel()[iPar]
            #ratio=1.+np.std(g_t)
            #norm=np.max([np.abs(np.mean(g_t))
            #ratio=np.cov(g_t)/Pa[iPar,iPar]
            #print np.cov(g_t),Pa[iPar,iPar],ratio
            ratio=np.abs(ThisG-np.mean(g_t))/np.sqrt(Pa[iPar,iPar]+Q[iPar,iPar])
            
            #ratio=np.abs(g_t[-1]-np.mean(g_t))/np.sqrt(Pa[iPar,iPar])#+Q[iPar,iPar]))
            diff=np.sum(w*(ThisG-g_t))/np.sum(w)
            ratio=np.abs(diff)/np.sqrt(Pa[iPar,iPar]+Q[iPar,iPar])
            F[iPar]=1.#ratio#/np.sqrt(2.)

            diff=ThisG-g_t[-1]#np.sum(w*(ThisG-g_t))/np.sum(w)
            #PaOut[iPar,iPar]=np.abs(diff)**2+Pa[iPar,iPar]+Q[iPar,iPar]
            PaOut[iPar,iPar]=np.abs(diff)**2+Pa[iPar,iPar]+Q[iPar,iPar]

        # Q=np.diag(np.ones((PaOut.shape[0],)))*(self.sigQ**2)

        PaOut=F.reshape((NPars,1))*Pa*F.reshape((1,NPars)).conj()+Q
        #print(F)
        #print(Q)
        # stop
        return PaOut
        


    def Evolve(self,xEst,Pa,CurrentTime):
        done=NpShared.GiveArray("%sSolsArray_done"%self.IdSharedMem)
        indDone=np.where(done==1)[0]
        Q=NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem)[self.iAnt]

        t0=NpShared.GiveArray("%sSolsArray_t0"%self.IdSharedMem)[indDone]
        t1=NpShared.GiveArray("%sSolsArray_t1"%self.IdSharedMem)[indDone]
        tm=NpShared.GiveArray("%sSolsArray_tm"%self.IdSharedMem)[indDone]


        G=NpShared.GiveArray("%sSolsArray_G"%self.IdSharedMem)[indDone][:,self.iAnt,:,:,:]

        
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


        F=np.ones((NPars,),G.dtype)
        if self.DoEvolve:
            if self.WeightType=="exp":
                w=np.exp(-tm0/self.WeigthScale)
                w/=np.sum(w)
                w=w[::-1]
            dx=1e-6
            for iPar in range(NPars):
                g_t=G[:,iPar]
                g_r=g_t.real.copy()
                g_i=g_t.imag.copy()
                
                ####
                z_r0 = np.polyfit(tm0, g_r, self.order, w=w)
                z_i0 = np.polyfit(tm0, g_i, self.order, w=w)
                poly_r = np.poly1d(z_r0)
                poly_i = np.poly1d(z_i0)
                x0_r=poly_r(ThisTime)
                x0_i=poly_i(ThisTime)
                Gout[iPar]=x0_r+1j*x0_i
                
                ####
                g_r[-1]+=dx
                g_i[-1]+=dx
                z_r1 = np.polyfit(tm0, g_r, self.order, w=w)
                z_i1 = np.polyfit(tm0, g_i, self.order, w=w)
                poly_r = np.poly1d(z_r1)
                poly_i = np.poly1d(z_i1)
                x1_r=poly_r(ThisTime)
                x1_i=poly_i(ThisTime)
                
                # dz=((x0_r-x1_r)+1j*(x0_i-x1_i))/dx

                xlast=G[-1][iPar]
                dz=((x0_r-xlast.real)+1j*(x0_i-xlast.imag))/np.sqrt((Pa[iPar,iPar]+Q[iPar,iPar]))
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
            #     print F
            
        # if self.iAnt==0:
        #     pylab.clf()
        #     pylab.imshow(np.diag(F).real,interpolation="nearest")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)


        
        #Pa=P[self.iAnt]
        PaOut=np.zeros_like(Pa)
        #Q=np.diag(np.ones((PaOut.shape[0],)))*(self.sigQ**2)
        PaOut=F.reshape((NPars,1))*Pa*F.reshape((1,NPars)).conj()+Q
        
        
        
        Gout=Gout.reshape((nd,npol,npol))
        print(np.diag(PaOut))

        return Gout,PaOut
  
