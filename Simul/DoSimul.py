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
#!/usr/bin/env python
import optparse
import sys
from killMS.Other import MyPickle
from killMS.Other import logo
from killMS.Other import ModColor
from killMS.Other import MyLogger
log=MyLogger.getLogger("killMS")
MyLogger.itsLog.logger.setLevel(MyLogger.logging.CRITICAL)
from killMS.Other import ClassTimeIt
from killMS.Data import ClassVisServer
from killMS.Predict.PredictGaussPoints_NumExpr import ClassPredict
from killMS.Predict.PredictGaussPoints_NumExpr5 import ClassPredict as ClassPredict5
from killMS.Array import ModLinAlg
from killMS.Array import NpShared
import time
import os
import numpy as np
import pickle
from SkyModel.Sky import ClassSM
from pyrap.tables import table
import glob


def main(options=None):
    #MSName="0000.MS"
    #SMName="MultiFreqs2.restored.corr.pybdsm.point.sky_in.npy"
    #ll=sorted(glob.glob("000?.point.w0.MS"))
    #SMName="Model2.txt.npy"
    #SMName="Model1_center.txt.npy"
    SMName="ModelRandom00.one.txt.npy"
    #SMName="ModelRandom00.gauss.txt.npy"
    #SMName="ModelRandom00.4.txt.npy"
    SMName="ModelRandom00.one.txt.npy"
    SMName="ModelRandom00.txt.npy"
    #SMName="ModelRandom00.25.txt.npy"
    #SMName="ModelRandom00.49.txt.npy"
    #SMName="ModelRandom00.txt.npy"
    #SMName="ModelSimulOne.txt.npy"
    #SMName="Deconv.Corr.npy"
    #ll=sorted(glob.glob("Simul.MS"))

    SMName="ModelRandom00.txt.npy"
    SMName="ModelRandom00.oneOff.txt.npy"
    #SMName="ModelRandom00.oneCenter.txt.npy"
    SMName="ModelImage.txt.npy"
    SMName="ModelRandom00.txt.npy"

    ll=sorted(glob.glob("000?.MS"))
    #ll=sorted(glob.glob("0000.MS"))
    #ll=sorted(glob.glob("BOOTES24_SB100-109.2ch8s.ms.tsel"))
    

    #ll=sorted(glob.glob("BOOTES24_SB100-109.2ch8s.ms"))

    #ll=sorted(glob.glob("0000.MS"))
    #ll=sorted(glob.glob("SimulHighRes.MS_p0"))
    #ll=sorted(glob.glob("SimulLowRes.MS_p0"))
    
    CS=ClassSimul(ll[0],SMName)
    Sols=CS.GiveSols()
    for l in ll:
        #CS=ClassSimul(l,SMName,Sols=Sols,ApplyBeam=True)
        CS=ClassSimul(l,SMName,Sols=Sols,ApplyBeam=False)
        CS.DoSimul()

class ClassSimul():

    def __init__(self,MSName,SMName,Sols=None,ApplyBeam=True):
        self.MSName=MSName
        self.SMName=SMName
        self.Sols=Sols
        self.ApplyBeam=ApplyBeam
        self.ChanMap=None
        self.Init()
        

    def GiveSols(self):
        MS=self.MS
        SM=self.SM
        VS=self.VS
        ApplyBeam=self.ApplyBeam
        na=MS.na
        nd=SM.NDir

        ###############################
        # NSols=80
        # tmin,tmax=MS.F_times.min(),MS.F_times.max()
        # tt=np.linspace(tmin,tmax,NSols+1)
        # Sols=np.zeros((NSols,),dtype=[("t0",np.float64),("t1",np.float64),("tm",np.float64),("G",np.complex64,(na,nd,2,2))])
        # Sols=Sols.view(np.recarray)
        # Sols.G[:,:,:,0,0]=1#e-3
        # Sols.G[:,:,:,1,1]=1#e-3
        # Sols.t0=tt[0:-1]
        # Sols.t1=tt[1::]
        # Sols.tm=(tt[0:-1]+tt[1::])/2.
        ###############################

        
        NSols=MS.F_ntimes

        nch=MS.NSPWChan


        Sols=np.zeros((NSols,),dtype=[("t0",np.float64),("t1",np.float64),("tm",np.float64),("G",np.complex64,(nch,na,nd,2,2))])
        Sols=Sols.view(np.recarray)
        Sols.G[:,:,:,:,0,0]=1 #e-3
        Sols.G[:,:,:,:,1,1]=1 #e-3
    
        dt=MS.dt
        Sols.t0=MS.F_times-dt/2.
        Sols.t1=MS.F_times+dt/2.
        Sols.tm=MS.F_times




        DeltaT_Amp=np.random.randn(nch,na,nd)*60
        period_Amp=120+np.random.randn(nch,na,nd)*10
        Amp_Mean=.9+np.random.rand(nch,na,nd)*0.2
        
        Amp_Amp=np.random.randn(nch,na,nd)

        Amp_Mean.fill(1.)
        Amp_Amp.fill(0.)
    
        DeltaT_Phase=np.random.randn(nch,na,nd)*60
        period_Phase=300+np.random.randn(nch,na,nd)*10
        #period_Phase=np.random.randn(na,nd)*10
        PhaseAbs=np.random.randn(nch,na,nd)*np.pi*0.3
        Amp_Phase=np.random.rand(nch,na,nd)*np.pi*0.5
    
        #Amp_Amp=np.zeros((na,nd))
        #PhaseAbs.fill(0)
        #Amp_Phase.fill(0)
        #Amp_Phase=np.zeros((na,nd))
        



        for itime in range(0,NSols):
            # if itime>0: 
            #     continue
            #continue
            print itime,"/",NSols
            for ich in range(nch):
                for iAnt in range(na):
                    for iDir in range(nd):
                        t=Sols.tm[itime]
                        t0=Sols.tm[0]
                        A=Amp_Mean[ich,iAnt,iDir]+Amp_Amp[ich,iAnt,iDir]*np.sin(DeltaT_Amp[ich,iAnt,iDir]+(t-t0)/period_Amp[ich,iAnt,iDir])
                        Phase=PhaseAbs[ich,iAnt,iDir]+Amp_Phase[ich,iAnt,iDir]*np.sin(DeltaT_Phase[ich,iAnt,iDir]+(t-t0)/period_Phase[ich,iAnt,iDir])
                        g0=A*np.exp(1j*Phase)
                        Sols.G[itime,ich,iAnt,iDir,0,0]=g0
                        #Sols.G[itime,iAnt,iDir,1,1]=g0

        ###############################

        DeltaT_Amp=np.random.randn(nch,na,nd)*60
        period_Amp=120+np.random.randn(nch,na,nd)*10
        Amp_Amp=np.random.randn(nch,na,nd)*.1
        Amp_Mean=np.random.rand(nch,na,nd)*2
   
        DeltaT_Phase=np.random.randn(nch,na,nd)*60
        period_Phase=300+np.random.randn(nch,na,nd)*10
        #period_Phase=np.random.randn(na,nd)*10
        PhaseAbs=np.random.randn(nch,na,nd)*np.pi
        Amp_Phase=np.random.randn(nch,na,nd)*np.pi#*0.1
    
        #Amp_Amp=np.zeros((na,nd))
        #PhaseAbs.fill(0)
        #Amp_Phase=np.zeros((na,nd))
    
        #print "!!!!!!!!!!!!!!!!!!!!! A=1"
        #print "!!!!!!!!!!!!!!!!!!!!! A=1"
        #print "!!!!!!!!!!!!!!!!!!!!! A=1"


        for itime in range(0,NSols):
            print "skip pol2"
            continue
            for ich in range(nch):
                for iAnt in range(na):
                    for iDir in range(nd):
                        t=Sols.tm[itime]
                        t0=Sols.tm[0]
                        A=Amp_Mean[ich,iAnt,iDir]+Amp_Amp[ich,iAnt,iDir]*np.sin(DeltaT_Amp[ich,iAnt,iDir]+(t-t0)/period_Amp[ich,iAnt,iDir])
                        Phase=PhaseAbs[ich,iAnt,iDir]+Amp_Phase[ich,iAnt,iDir]*np.sin(DeltaT_Phase[ich,iAnt,iDir]+(t-t0)/period_Phase[ich,iAnt,iDir])
                        g0=A*np.exp(1j*Phase)
                        Sols.G[itime,ich,iAnt,iDir,1,1]=g0
                        #Sols.G[itime,iAnt,iDir,1,1]=g0



        # # Equalise in time
        # for itime in range(NSols):
        #     Sols.G[itime,:,:,:,0,0]=Sols.G[0,:,:,:,0,0]
        #     Sols.G[itime,:,:,:,1,1]=Sols.G[0,:,:,:,1,1]

        # equalise in freq
        for ich in range(1,nch):
            Sols.G[:,ich,:,:,:,:]=Sols.G[:,0,:,:,:,:]


        # make scalar
        Sols.G[:,:,:,:,1,1]=Sols.G[:,:,:,:,0,0]

        # unity
        Sols.G.fill(0)
        Sols.G[:,:,:,:,0,0]=1.
        Sols.G[:,:,:,:,1,1]=1.


        # # Sols.G[:,:,:,1:,0,0]=0.01
        # # Sols.G[:,:,:,1:,1,1]=0.01

        return Sols

    def GiveJones(self):
        if self.Sols is None:
            Sols=self.GiveSols()
        else:
            Sols=self.Sols

        MS=self.MS
        SM=self.SM
        VS=self.VS
        ApplyBeam=self.ApplyBeam
        na=MS.na
        nd=SM.NDir

        Jones={}
        Jones["t0"]=Sols.t0
        Jones["t1"]=Sols.t1
    
        nt,nch,na,nd,_,_=Sols.G.shape
        G=np.swapaxes(Sols.G,1,3).reshape((nt,nd,na,nch,2,2))


        # G[:,:,:,:,0,0]/=np.abs(G[:,:,:,:,0,0])
        # G[:,:,:,:,1,1]=G[:,:,:,:,0,0]

        # G.fill(0)
        # G[:,:,:,:,0,0]=1
        # G[:,:,:,:,1,1]=1

        nt,nd,na,nch,_,_=G.shape
#        G=np.random.randn(*G.shape)+1j*np.random.randn(*G.shape)
    
    
        useArrayFactor=True
        useElementBeam=False
        if ApplyBeam:
            print ModColor.Str("Apply Beam")
            MS.LoadSR(useElementBeam=False,useArrayFactor=True)
            RA=SM.ClusterCat.ra
            DEC=SM.ClusterCat.dec
            NDir=RA.size
            Tm=Sols.tm
            T0s=Sols.t0
            T1s=Sols.t1
            DicoBeam={}
            DicoBeam["Jones"]=np.zeros((Tm.size,NDir,MS.na,MS.NSPWChan,2,2),dtype=np.complex64)
            DicoBeam["t0"]=np.zeros((Tm.size,),np.float64)
            DicoBeam["t1"]=np.zeros((Tm.size,),np.float64)
            DicoBeam["tm"]=np.zeros((Tm.size,),np.float64)
    
            rac,decc=MS.radec

            for itime in range(Tm.size):
                print itime
                DicoBeam["t0"][itime]=T0s[itime]
                DicoBeam["t1"][itime]=T1s[itime]
                DicoBeam["tm"][itime]=Tm[itime]
                ThisTime=Tm[itime]
                Beam=MS.GiveBeam(ThisTime,RA,DEC)

                ###### Normalise
                Beam0=MS.GiveBeam(ThisTime,np.array([rac]),np.array([decc]))
                Beam0inv=ModLinAlg.BatchInverse(Beam0)
                nd,_,_,_,_=Beam.shape
                Ones=np.ones((nd, 1, 1, 1, 1),np.float32)
                Beam0inv=Beam0inv*Ones
                Beam=ModLinAlg.BatchDot(Beam0inv,Beam)
                ######


                DicoBeam["Jones"][itime]=Beam
                
            nt,nd,na,nch,_,_= DicoBeam["Jones"].shape

            #m=np.mean(np.abs(DicoBeam["Jones"][:,1,:,:,:,:]))
            # m=np.mean(np.abs(DicoBeam["Jones"][:,1,:,:,:,:]))
            # DicoBeam["Jones"][:,1,0:6,:,:,:]*=2
            # DicoBeam["Jones"][:,1,:,:,:,:]/=np.mean(np.abs(DicoBeam["Jones"][:,1,:,:,:,:]))
            # DicoBeam["Jones"][:,1,:,:,:,:]*=m


            # #################"
            # # Single Channel
            # DicoBeam["Jones"]=np.mean(DicoBeam["Jones"],axis=3).reshape((nt,nd,na,1,2,2))

            # G=ModLinAlg.BatchDot(G,DicoBeam["Jones"])

            # #################"
            # Multiple Channel
            Ones=np.ones((1, 1, 1, nch, 1, 1),np.float32)
            G=G*Ones
            G=ModLinAlg.BatchDot(G,DicoBeam["Jones"])

            # #################"

            # G[:,:,:,:,0,0]=1
            # G[:,:,:,:,0,1]=0.5
            # G[:,:,:,:,1,0]=2.
            # G[:,:,:,:,1,1]=1





            print "Done"
    
    
        # #################
        # Multiple Channel
        self.ChanMap=range(nch)
        # #################
    
        Jones["Beam"]=G
        Jones["BeamH"]=ModLinAlg.BatchH(G)
        if self.ChanMap is None:
            self.ChanMap=np.zeros((VS.MS.NSPWChan,),np.int32).tolist()
        
        Jones["ChanMap"]=self.ChanMap


        # ###### for PM5
        # Jones["Map_VisToJones_Freq"]=self.ChanMap
        # Jones["Jones"]=Jones["Beam"]
        # nt=VS.MS.times_all.size
        # ntJones=DicoBeam["tm"].size
        # d=VS.MS.times_all.reshape((nt,1))-DicoBeam["tm"].reshape((1,ntJones))
        # Jones["Map_VisToJones_Time"]=np.argmin(np.abs(d),axis=1)

        return Jones
    
    
    def Init(self):
        ReadColName="DATA"
        WriteColName="DATA"
        SM=ClassSM.ClassSM(self.SMName)
        #SM.Type="Catalog"

        VS=ClassVisServer.ClassVisServer(self.MSName,ColName=ReadColName,
                                         TVisSizeMin=1,
                                         TChunkSize=14)
        self.VS=VS
        VS.setSM(SM)
        VS.CalcWeigths()
        MS=VS.MS
        SM.Calc_LM(MS.rac,MS.decc)
        print MS
        MS.PutBackupCol(incol="CORRECTED_DATA")

        self.MS=MS
        self.SM=SM
        # SM.SourceCat.l[:]=-0.009453866781636
        # SM.SourceCat.m[:]=0.009453866781636
        # stop


    def DoSimul(self):
    
        Noise=0.
        MS=self.MS
        SM=self.SM
        VS=self.VS
        ApplyBeam=self.ApplyBeam
        na=MS.na
        nd=SM.NDir
        NCPU=6

        #PM=ClassPredict(NCPU=NCPU,DoSmearing="F")
        PM=ClassPredict(NCPU=NCPU)
        PM5=ClassPredict5(NCPU=NCPU)
        na=MS.na
        nd=SM.NDir
        
        Load=VS.LoadNextVisChunk()
    
        Jones = self.GiveJones()
    
    
        print>>log, ModColor.Str("Substract sources ... ",col="green")
        #SM.SelectSubCat(SM.SourceCat.kill==0)


        PredictData=PM.predictKernelPolCluster(VS.ThisDataChunk,SM,ApplyTimeJones=Jones,Noise=Noise)
        # PredictData=PM5.predictKernelPolCluster(VS.ThisDataChunk,SM,ApplyTimeJones=Jones)

        # import pylab
        # ind=np.where((VS.ThisDataChunk["A0"]==7)&(VS.ThisDataChunk["A1"]==17))[0]
        # op0=np.real
        # op1=np.abs
        # pylab.clf()
        # pylab.subplot(2,1,1)
        # pylab.plot(op0(PredictData[ind,0,0]))
        # pylab.plot(op0(PredictData5[ind,0,0]))
        # pylab.plot(op0(PredictData[ind,0,0])-op0(PredictData5[ind,0,0]))
        # pylab.subplot(2,1,2)
        # pylab.plot(op1(PredictData[ind,0,0]))
        # pylab.plot(op1(PredictData5[ind,0,0]))
        # pylab.plot(op1(PredictData[ind,0,0])-op1(PredictData5[ind,0,0]))
        # pylab.draw()
        # pylab.show(False)
        # stop
        # #PredictData=PM.predictKernelPolCluster(VS.ThisDataChunk,SM)
        
        #SM.RestoreCat()
    
        MS.data=PredictData
    
    
    
    
        #VS.MS.SaveVis(Col="DATA")
        #VS.MS.SaveVis(Col="CORRECTED_DATA")
        #VS.MS.SaveVis(Col="CORRECTED_DATA_BACKUP")
        VS.MS.SaveVis(Col="CORRECTED_DATA")

        # t=table(self.MSName,readonly=False)
        # f=t.getcol("FLAG")
        # f.fill(0)
        # r=np.random.rand(*(f.shape[0:2]))
        # ff=(r<0.3)
        # # indr,indf=np.where(ff)
        # # f[indr,indf,:]=True
        # # # MS.flag_all=f
        # # # MS.data[f]=1.e10
        # t.putcol("FLAG",f)
        # t.putcol("FLAG_BACKUP",f)
        # t.close()

        
        Sols=self.Sols
        FileName="Simul.npz"
        np.savez(FileName,Sols=Sols,StationNames=MS.StationNames,SkyModel=SM.ClusterCat,ClusterCat=SM.ClusterCat,
                 BeamTimes=np.array([],np.float64))
        #self.Plot()
        
    
    def Plot(self):

        from pyrap.tables import table
        t=table("BOOTES24_SB100-109.2ch8s.ms/",readonly=False)
        D0=t.getcol("MODEL_DATA")
        D1=t.getcol("CORRECTED_DATA_BACKUP")
        f=t.getcol("FLAG")
        DD=D0-D1
        DD[f==1]=0
        #inp=np.where(np.abs(DD)==np.max(np.abs(DD)))
        A0=t.getcol("ANTENNA1")
        A1=t.getcol("ANTENNA2")
        indA=np.where((A0==48)&(A1==55))[0]
        d0=D0[indA]
        d1=D1[indA]

        pylab.clf()
        pylab.plot(np.angle(d1[:,0,0])-np.angle(d0[:,0,0]))
        pylab.show()
