#!/usr/bin/env python
import optparse
import sys
from killMS2.Other import MyPickle
from killMS2.Other import logo
from killMS2.Other import ModColor
from killMS2.Other import MyLogger
log=MyLogger.getLogger("killMS")
MyLogger.itsLog.logger.setLevel(MyLogger.logging.CRITICAL)
from killMS2.Other import ClassTimeIt
from killMS2.Data import ClassVisServer
from killMS2.Predict.PredictGaussPoints_NumExpr import ClassPredict
from killMS2.Array import ModLinAlg
from killMS2.Array import NpShared
import time
import os
import numpy as np
import pickle
from SkyModel.Sky import ClassSM
from pyrap.tables import table
import glob


def main(options=None):
    MSName="0000.MS"
    #SMName="MultiFreqs2.restored.corr.pybdsm.point.sky_in.npy"
    #ll=sorted(glob.glob("000?.point.w0.MS"))
    SMName="ModelRandom00.txt.npy"
    ll=sorted(glob.glob("0000.MS"))
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
        self.Init()

    def GiveSols(self):
        MS=self.MS
        SM=self.SM
        VS=self.VS
        ApplyBeam=self.ApplyBeam
        na=MS.na
        nd=SM.NDir
        NSols=MS.F_ntimes
        Sols=np.zeros((NSols,),dtype=[("t0",np.float64),("t1",np.float64),("tm",np.float64),("G",np.complex64,(na,nd,2,2))])
        Sols=Sols.view(np.recarray)
        Sols.G[:,:,:,0,0]=1#e-3
        Sols.G[:,:,:,1,1]=1#e-3
    
        dt=MS.dt
        Sols.t0=MS.F_times-dt/2.
        Sols.t1=MS.F_times+dt/2.
        Sols.tm=MS.F_times


        DeltaT_Amp=np.random.randn(na,nd)*60
        period_Amp=120+np.random.randn(na,nd)*10
        Amp_Amp=np.random.randn(na,nd)*.1
    
        DeltaT_Phase=np.random.randn(na,nd)*60
        period_Phase=300+np.random.randn(na,nd)*10
        #period_Phase=np.random.randn(na,nd)*10
        PhaseAbs=np.random.randn(na,nd)*np.pi
        Amp_Phase=np.random.randn(na,nd)*np.pi#*0.1
    
        #Amp_Amp=np.zeros((na,nd))
        PhaseAbs.fill(0)
        #Amp_Phase=np.zeros((na,nd))
    
        for itime in range(0,NSols):
            for iAnt in range(na):
                for iDir in range(nd):
                    t=Sols.tm[itime]
                    t0=Sols.tm[0]
                    A=.5+Amp_Amp[iAnt,iDir]*np.sin(DeltaT_Amp[iAnt,iDir]+(t-t0)/period_Amp[iAnt,iDir])
                    Phase=PhaseAbs[iAnt,iDir]+Amp_Phase[iAnt,iDir]*np.sin(DeltaT_Phase[iAnt,iDir]+(t-t0)/period_Phase[iAnt,iDir])
                    g0=A*np.exp(1j*Phase)
                    Sols.G[itime,iAnt,iDir,0,0]=g0
                    #Sols.G[itime,iAnt,iDir,1,1]=g0

        ###############################

        DeltaT_Amp=np.random.randn(na,nd)*60
        period_Amp=120+np.random.randn(na,nd)*10
        Amp_Amp=np.random.randn(na,nd)*.1
    
        DeltaT_Phase=np.random.randn(na,nd)*60
        period_Phase=300+np.random.randn(na,nd)*10
        #period_Phase=np.random.randn(na,nd)*10
        PhaseAbs=np.random.randn(na,nd)*np.pi
        Amp_Phase=np.random.randn(na,nd)*np.pi#*0.1
    
        #Amp_Amp=np.zeros((na,nd))
        #PhaseAbs.fill(0)
        #Amp_Phase=np.zeros((na,nd))
    
        for itime in range(0,NSols):
            for iAnt in range(na):
                for iDir in range(nd):
                    t=Sols.tm[itime]
                    t0=Sols.tm[0]
                    A=.5+Amp_Amp[iAnt,iDir]*np.sin(DeltaT_Amp[iAnt,iDir]+(t-t0)/period_Amp[iAnt,iDir])
                    Phase=PhaseAbs[iAnt,iDir]+Amp_Phase[iAnt,iDir]*np.sin(DeltaT_Phase[iAnt,iDir]+(t-t0)/period_Phase[iAnt,iDir])
                    g0=A*np.exp(1j*Phase)
                    Sols.G[itime,iAnt,iDir,1,1]=g0
                    #Sols.G[itime,iAnt,iDir,1,1]=g0



        return Sols

    def GiveJones(self):
        if self.Sols==None:
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
    
        nt,na,nd,_,_=Sols.G.shape
        G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))

        G.fill(0)
        G[:,:,:,:,0,0]=1
        G[:,:,:,:,1,1]=1

        # G[:,:,:,:,0,1]=0.
        # G[:,:,:,:,1,0]=0.
    
    
        useArrayFactor=True
        useElementBeam=False
        if ApplyBeam:
            print ModColor.Str("Apply Beam")
            MS.LoadSR()
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
    
            for itime in range(Tm.size):
                DicoBeam["t0"][itime]=T0s[itime]
                DicoBeam["t1"][itime]=T1s[itime]
                DicoBeam["tm"][itime]=Tm[itime]
                ThisTime=Tm[itime]
                DicoBeam["Jones"][itime]=MS.GiveBeam(ThisTime,RA,DEC)
    
            nt,nd,na,nch,_,_= DicoBeam["Jones"].shape
            DicoBeam["Jones"]=np.mean(DicoBeam["Jones"],axis=3).reshape((nt,nd,na,1,2,2))
            G=ModLinAlg.BatchDot(G,DicoBeam["Jones"])
            print "Done"
    
    
    
        Jones["Beam"]=G
        Jones["BeamH"]=ModLinAlg.BatchH(G)
        Jones["ChanMap"]=np.zeros((VS.MS.NSPWChan,)).tolist()


        return Jones
    
    
    def Init(self):
        ReadColName="DATA"
        WriteColName="DATA"
        SM=ClassSM.ClassSM(self.SMName)
        VS=ClassVisServer.ClassVisServer(self.MSName,ColName=ReadColName,
                                         TVisSizeMin=1,
                                         TChunkSize=14)
        self.VS=VS
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
    
        Noise=0.0
        MS=self.MS
        SM=self.SM
        VS=self.VS
        ApplyBeam=self.ApplyBeam
        na=MS.na
        nd=SM.NDir
        NCPU=6

        PM=ClassPredict(NCPU=NCPU)
        na=MS.na
        nd=SM.NDir
    
        Load=VS.LoadNextVisChunk()
    
        Jones = self.GiveJones()
    
    
        print>>log, ModColor.Str("Substract sources ... ",col="green")
        #SM.SelectSubCat(SM.SourceCat.kill==0)
        PredictData=PM.predictKernelPolCluster(VS.ThisDataChunk,SM,ApplyTimeJones=Jones,Noise=Noise)
        
        #SM.RestoreCat()
    
        MS.data=PredictData
    
    
    
    
        #VS.MS.SaveVis(Col="DATA")
        #VS.MS.SaveVis(Col="CORRECTED_DATA")
        VS.MS.SaveVis(Col="CORRECTED_DATA_BACKUP")

        t=table(self.MSName,readonly=False)
        f=t.getcol("FLAG")
        f.fill(0)
        # r=np.random.rand(*(f.shape[0:2]))
        # ff=(r>0.7)
        # indr,indf=np.where(ff)
        # f[indr,indf,:]=True
        # # MS.flag_all=f
        # # MS.data[f]=1.e10
        t.putcol("FLAG",f)
        t.putcol("FLAG_BACKUP",f)
        t.close()

        
        Sols=self.Sols
        FileName="Simul.npz"
        np.savez(FileName,Sols=Sols,StationNames=MS.StationNames,SkyModel=SM.ClusterCat,ClusterCat=SM.ClusterCat)
        
        
    
