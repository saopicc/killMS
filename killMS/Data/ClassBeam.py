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
from . import ClassMS
from pyrap.tables import table
from DDFacet.Other import logger
log=logger.getLogger("ClassBeam")
from killMS.Array import ModLinAlg

class ClassBeam():
    def __init__(self,MSName,GD,SM):
        self.GD=GD
        self.SM=SM
        self.MSName=MSName#self.GD["VisData"]["MSName"]
        self.ColName=self.GD["VisData"]["InCol"]
        self.MS=ClassMS.ClassMS(self.MSName,Col=self.ColName,DoReadData=False,GD=GD)
        self.DtBeamMin=self.GD["Beam"]["DtBeamMin"]

    def GiveMeanBeam(self):
        log.print( "Calculate mean beam for covariance estimate ... ")
        t=table(self.MSName,ack=False)
        times=t.getcol("TIME")
        t.close()
        
        DicoBeam=self.GiveBeam(times)
        J=DicoBeam["Jones"]
        AbsMean=np.mean(np.abs(J),axis=0)
        return AbsMean
    # def SetLOFARBeam(self,LofarBeam):
    #     self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
    #     log.print( "Set LOFARBeam in %s Mode"%self.BeamMode)
    #     useArrayFactor=("A" in self.BeamMode)
    #     useElementBeam=("E" in self.BeamMode)
    #     self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
    #     self.ApplyBeam=True
        
        
    def GiveBeam(self,times):
        if self.GD["Beam"]["BeamModel"]=="LOFAR":
            useArrayFactor=("A" in self.GD["Beam"]["LOFARBeamMode"])
            useElementBeam=("E" in self.GD["Beam"]["LOFARBeamMode"])
            self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
        elif self.GD["Beam"]["BeamModel"]=="FITS":
            self.MS.LoadFITSBeam()

        #log.print( "  Update beam [Dt = %3.1f min] ... "%self.DtBeamMin)
        DtBeamSec=self.DtBeamMin*60
        tmin,tmax=times[0],times[-1]
        TimesBeam=np.arange(tmin,tmax,DtBeamSec).tolist()
        if not(tmax in TimesBeam): TimesBeam.append(tmax)
        TimesBeam=np.array(TimesBeam)
        T0s=TimesBeam[:-1]
        T1s=TimesBeam[1:]
        Tm=(T0s+T1s)/2.
        RA,DEC=self.SM.ClusterCat.ra,self.SM.ClusterCat.dec
        NDir=RA.size
        Beam=np.zeros((Tm.size,NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)
        for itime in range(Tm.size):
            ThisTime=Tm[itime]
            Beam[itime]=self.MS.GiveBeam(ThisTime,RA,DEC)
    
        ###### Normalise
        rac,decc=self.MS.radec
        if self.GD["Beam"]["CenterNorm"]==1:
            for itime in range(Tm.size):
                ThisTime=Tm[itime]
                Beam0=self.MS.GiveBeam(ThisTime,np.array([rac]),np.array([decc]))
                Beam0inv=ModLinAlg.BatchInverse(Beam0)
                nd,_,_,_,_=Beam[itime].shape
                Ones=np.ones((nd, 1, 1, 1, 1),np.float32)
                Beam0inv=Beam0inv*Ones
                Beam[itime]=ModLinAlg.BatchDot(Beam0inv,Beam[itime])
        ###### 

        nt,nd,na,nch,_,_= Beam.shape
        Beam=np.mean(Beam,axis=3).reshape((nt,nd,na,1,2,2))
        
        
        DicoBeam={}
        DicoBeam["t0"]=T0s
        DicoBeam["t1"]=T1s
        DicoBeam["tm"]=Tm
        DicoBeam["Jones"]=Beam
        return DicoBeam
