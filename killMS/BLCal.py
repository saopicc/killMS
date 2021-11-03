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



import matplotlib.pyplot as pylab
from pyrap.tables import table
from killMS.Data import ClassWeighting
from DDFacet.Other import logger
log=logger.getLogger("ClassVisServer")
import numpy as np

def test():
    MSName="BOOTES24_SB100-109.2ch8s.ms"
    BLC=ClassBLCal(MSName)
    BLC.CalcBLCal()

class ClassBLCal():
    def __init__(self,MSName):
        self.MSName=MSName
        self.Init()

    def Init(self):
        MSName=self.MSName
        log.print( "Reading %s"%MSName)
        t=table(MSName,readonly=False,ack=False)
        self.D=t.getcol("CORRECTED_DATA_BACKUP")
        self.P=t.getcol("PREDICTED_DATA")
        self.Time=t.getcol("TIME")
        self.A0=t.getcol("ANTENNA1")
        self.A1=t.getcol("ANTENNA2")
        self.f=t.getcol("FLAG")
        self.uvw=t.getcol("UVW")
        t.close()

        t=table("%s/SPECTRAL_WINDOW"%MSName,readonly=False,ack=False)
        self.Freqs=t.getcol("CHAN_FREQ").flatten()
        t.close()

        self.na=np.max([np.max(self.A1),np.max(self.A0)])+1


        #
        self.CalcWeights()

    def CalcWeights(self):
        
        flags=self.f
        uvw=self.uvw
        Freqs=self.Freqs
        
        u,v,w=uvw.T
        freq=np.mean(Freqs)
        uvmax=np.max(np.sqrt(u**2+v**2))
        FOV=5.
        res=uvmax*freq/3.e8
        npix=(FOV*np.pi/180)/res
        ImShape=(1,1,npix,npix)
        Robust=-0.5
        VisWeights=np.ones((uvw.shape[0],Freqs.size),dtype=np.float32)
        
        WeightMachine=ClassWeighting.ClassWeighting(ImShape,res)
        VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,flags,Freqs,
                                             Robust=Robust,
                                             Weighting="Briggs")
        VisWeights/=np.max(VisWeights)
        self.VisWeights=VisWeights


    def CalcBLCal(self):
        nt=1
        Time=self.Time
        GdTime=np.linspace(Time[0],Time[-1]+1,nt+1)
        t0=GdTime[0:-1]
        t1=GdTime[1::]
        nrows,nch,_=self.D.shape
        na=self.na
        D=self.D
        P=self.P
        A0=self.A0
        A1=self.A1
        f=self.f
        VisWeights=self.VisWeights

        C=np.zeros((nt,nch,na,na),dtype=np.float32)
        for iAnt in range(na):
            print("[%i/%i] -"%(iAnt,na))
            for jAnt in range(iAnt,na):
                if iAnt==jAnt: continue
                c0=((A0==iAnt)&(A1==jAnt))
                ind=np.where(c0)[0]
                
                if ind.size==0: continue
                Dsel=D[ind,:,:]
                Psel=P[ind,:,:]
                fsel=f[ind,:,:]
                Wsel=VisWeights[ind,:]
                Tsel=Time[ind]
                
                for it in range(nt):
                    indt=np.where((Tsel>=t0[it])&(Tsel<t1[it]))[0]
                    Dsel_t=Dsel[indt,:,:]
                    Psel_t=Psel[indt,:,:]
                    fsel_t=fsel[indt,:,:]
                    Wsel_t=Wsel[indt,:]
                    #print indt.size
                    
                    for ich in range(nch):
                        Dsel_t_ch=Dsel_t[:,ich,0]
                        Psel_t_ch=Psel_t[:,ich,0]
                        fsel_t_ch=fsel_t[:,ich,0]
                        Wsel_t_ch=Wsel_t[:,ich]
                        indf=np.where(fsel_t_ch==0)[0]
                        
                        d=Dsel_t_ch[indf]
                        p=Psel_t_ch[indf]
                        if indf.size==0: continue
                        w=np.sum(Wsel_t_ch[indf])/float(indf.size)
                        
                        if indf.size==0: continue
                        C[it,ich,iAnt,jAnt]=1.-w+w*0.5*np.mean(d*p.conj()+p*d.conj())/np.mean(p*p.conj())

        C[np.isnan(C)]=0
        C[C==0]=1
        np.save("C",C)


    def Plot(self):
        ich=0
        it=0
        pylab.figure(figsize=(18,8))
        pylab.subplot(1,2,1)
        pylab.imshow(np.abs(C[it,ich]),interpolation="nearest")
        pylab.xlabel("Antenna number")
        pylab.ylabel("Antenna number")
        pylab.colorbar()
        pylab.subplot(1,2,2)
        pylab.imshow(np.angle(C[it,ich]),interpolation="nearest")
        pylab.xlabel("Antenna number")
        pylab.ylabel("Antenna number")
        pylab.colorbar()


    def Apply(self):


        C=np.load("C.npy")
        for iAnt in range(na):
            print("[%i/%i] -"%(iAnt,na))
            for jAnt in range(iAnt,na):
                if iAnt==jAnt: continue
                c0=((A0==iAnt)&(A1==jAnt))
                ind=np.where(c0)[0]
                Tsel=Time[ind]
                for it in range(nt):
                    indt=np.where((Tsel>=t0[it])&(Tsel<t1[it]))[0]
                    indTime=ind[indt]
                    for ich in range(nch):
                        P[indTime,ich,:]*=np.abs(C[it,ich,iAnt,jAnt])


    def ToMS(self):
        D-=P

        t=table(self.MSName,readonly=False,ack=False)
        t.putcol("CORRECTED_DATA",D)
        t.close()
