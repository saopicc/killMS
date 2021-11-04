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

from killMS.Data import ClassVisServer
#from Sky import ClassSM
from killMS.Array import ModLinAlg
#import matplotlib.pyplot as pylab

from DDFacet.Other import logger
log=logger.getLogger("ClassWirtingerSolver")
from killMS.Other import ModColor

#from killMS.Other.progressbar import ProgressBar
from DDFacet.Other.progressbar import ProgressBar
            
#from Sky.PredictGaussPoints_NumExpr import ClassPredict
from killMS.Other import ClassTimeIt
from killMS.Other import Counter
from .ClassEvolve import ClassModelEvolution
import time
from itertools import product as ItP
from killMS.Wirtinger import ClassSolverLM
from killMS.Wirtinger import ClassSolverEKF
from killMS.Wirtinger import ClassSolPredictMachine
from SkyModel.Sky import ClassSM

def test():


    ReadColName="DATA"
    WriteColName="CORRECTED_DATA"
    SM=ClassSM.ClassSM("/media/tasse/data/HyperCal2/test/ModelRandom00.txt.npy",
                       killdirs=["c0s0."],invert=False)
    
    VS=ClassVisServer.ClassVisServer("/media/6B5E-87D0/MS/SimulTec/Pointing00/MS/0000.MS",ColName=ReadColName,
                                     TVisSizeMin=2,
                                     TChunkSize=.1)
    
    #LM=ClassWirtingerSolver(VS,SM,PolMode="Scalar",NIter=1,SolverType="EKF")#20)
    LM=ClassWirtingerSolver(VS,SM,PolMode="Scalar",NIter=10,SolverType="KAFCA",evP_StepStart=3, evP_Step=10)#"CohJones")#"KAFCA")
    # LM=ClassWirtingerSolver(VS,SM,PolMode="Scalar",NIter=10,SolverType="CohJones")#"KAFCA")
    # LM.doNextTimeSolve()
    # LM.doNextTimeSolve_Parallel()
    # return
    PM=ClassPredict()
    SM=LM.SM
    LM.InitSol()

    VS.LoadNextVisChunk()

    while True:
        Res=LM.setNextData()
        if Res==True:
            #print Res,VS.CurrentVisTimes_SinceStart_Minutes
            LM.doNextTimeSolve_Parallel()
            #LM.doNextTimeSolve()
            continue
        else:
            # substract
            pass
            # Jones={}
            # Jones["t0"]=LM.Sols.t0
            # Jones["t1"]=LM.Sols.t1
            # nt,na,nd,_,_=LM.Sols.G.shape
            # G=np.swapaxes(LM.Sols.G,1,2).reshape((nt,nd,na,1,2,2))
            # Jones["Beam"]=G
            # Jones["BeamH"]=ModLinAlg.BatchH(G)

            # SM.SelectSubCat(SM.SourceCat.kill==1)
            # PredictData=PM.predictKernelPolCluster(LM.VS.ThisDataChunk,LM.SM,ApplyTimeJones=Jones)
            # SM.RestoreCat()

            # LM.VS.ThisDataChunk["data"]-=PredictData
            # LM.VS.MS.data=LM.VS.ThisDataChunk["data"]
            # LM.VS.MS.SaveVis(Col=WriteColName)

        if Res=="EndChunk":
            Load=VS.LoadNextVisChunk()
            if Load=="EndOfObservation":
                break


        
    # import pylab
    # t=np.array(VS.TEST_TLIST)
    # dt=t[1::]-t[0:-1]
    # pylab.clf()
    # pylab.plot(dt)
    # pylab.draw()
    # pylab.show(False)

class ClassWirtingerSolver():

    def __init__(self,VS,SM,
                 BeamProps=None,
                 PolMode="IFull",
                 Lambda=1,NIter=20,
                 NCPU=6,
                 SolverType="CohJones",
                 IdSharedMem="",
                 evP_StepStart=0, evP_Step=1,
                 DoPlot=False,
                 DoPBar=True,GD=None,
                 ConfigJacobianAntenna={},
                 TypeRMS="GlobalData",
                 VS_PredictCol=None):
        self.VS_PredictCol=VS_PredictCol
        self.DType=np.complex128
        self.TypeRMS=TypeRMS
        self.IdSharedMem=IdSharedMem
        self.ConfigJacobianAntenna=ConfigJacobianAntenna
        self.Lambda=Lambda
        self.NCPU=NCPU
        self.DoPBar=DoPBar
        self.GD=GD
        self.Q=None
        self.PListKeep=[]
        self.QListKeep=[]

        # if BeamProps!=None:
        #     rabeam,decbeam=SM.ClusterCat.ra,SM.ClusterCat.dec
        #     Mode,TimeMin=BeamProps
        #     LofarBeam=(Mode,TimeMin,rabeam,decbeam)
        #     VS.SetBeam(LofarBeam)

        MS=VS.MS
        if SM.Type=="Catalog":
            SM.Calc_LM(MS.rac,MS.decc)
        self.SM=SM
        self.VS=VS
        self.DoPlot=DoPlot
        if DoPlot==2:
            self.InitPlotGraph()
        self.PolMode=PolMode

        self.SM_Compress=None
        if (self.GD["Compression"]["CompressionMode"] is not None) or self.GD["Compression"]["CompressionDirFile"]:
            log.print(ModColor.Str("Using compression with Mode = %s"%self.GD["Compression"]["CompressionMode"]))
            
            if self.GD["Compression"]["CompressionMode"] and self.GD["Compression"]["CompressionMode"].lower()=="auto":
                ClusterCat=SM.ClusterCat#[1:2]
                #self.SM_Compress=ClassSM.ClassSM(SM)
                
            else:
                ClusterCat=np.load(self.GD["Compression"]["CompressionDirFile"])
                
            ClusterCat=ClusterCat.view(np.recarray)
            SourceCat=np.zeros((ClusterCat.shape[0],),dtype=[('Name', 'S200'), ('ra', '<f8'), ('dec', '<f8'), ('Sref', '<f8'),
                                                             ('I', '<f8'), ('Q', '<f8'), ('U', '<f8'), ('V', '<f8'), ('RefFreq', '<f8'),
                                                             ('alpha', '<f8'), ('ESref', '<f8'), ('Ealpha', '<f8'), ('kill', '<i8'),
                                                             ('Cluster', '<i8'), ('Type', '<i8'), ('Gmin', '<f8'), ('Gmaj', '<f8'),
                                                             ('Gangle', '<f8'), ('Select', '<i8'), ('l', '<f8'), ('m', '<f8'), ('Exclude', '<i8')])
            SourceCat=SourceCat.view(np.recarray)
            SourceCat.ra[:]=ClusterCat.ra[:]
            SourceCat.dec[:]=ClusterCat.dec[:]
            SourceCat.RefFreq[:]=100.e6
            SourceCat.I[:]=1.
            SourceCat.Cluster=np.arange(ClusterCat.shape[0])
            np.save("SM_Compress.npy",SourceCat)
            self.SM_Compress=ClassSM.ClassSM("SM_Compress.npy")
            self.SM_Compress.Calc_LM(self.VS.MS.rac,self.VS.MS.decc)
                        
                
        if self.PolMode=="IDiag":
            npolx=2
            npoly=1
        elif self.PolMode=="Scalar":
            npolx=1
            npoly=1
        elif self.PolMode=="IFull":
            npolx=2
            npoly=2

        self.NJacobBlocks_X,self.NJacobBlocks_Y=npolx,npoly

        self.SolPredictMachine=None
        if self.GD["KAFCA"]["EvolutionSolFile"]!="":
            self.SolPredictMachine=ClassSolPredictMachine.ClassSolPredictMachine(GD)
            
        
        self.G=None
        self.NIter=NIter
        #self.SolsList=[]
        self.iCurrentSol=0
        self.SolverType=SolverType
        self.rms=None
        self.rmsFromData=None
        self.SM.ApparentSumI=None
        # if SolverType=="KAFCA":
        #     log.print( ModColor.Str("niter=%i"%self.NIter))
        #     #self.NIter=1
        self.EvolvePStepStart,EvolvePStep=evP_StepStart,evP_Step
        self.CounterEvolveP=Counter.Counter(EvolvePStep)
        self.ThisStep=0
        self.rmsFromExt=None
    # def AppendEmptySol(self):
    #     #### Solutions
    #     # self.NSols=self.VS.TimesVisMin.size-1
    #     na=self.VS.MS.na
    #     nd=self.SM.NDir
    #     Sol=np.zeros((1,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(na,nd,2,2))])
    #     self.SolsList.append(Sol.view(np.recarray))

    def GiveSols(self,SaveStats=False):
        ind=np.where(self.SolsArray_done==1)[0]
        self.SolsArray_Full.t0[0:ind.size]=self.SolsArray_t0[0:ind.size]
        self.SolsArray_Full.t1[0:ind.size]=self.SolsArray_t1[0:ind.size]
        self.SolsArray_Full.Stats[0:ind.size]=self.SolsArray_Stats[0:ind.size]
        if self.PolMode=="Scalar":
            self.SolsArray_Full.G[0:ind.size,:,:,:,0,0]=self.SolsArray_G[0:ind.size,:,:,:,0,0]
            self.SolsArray_Full.G[0:ind.size,:,:,:,1,1]=self.SolsArray_G[0:ind.size,:,:,:,0,0]
        elif self.PolMode=="IDiag":
            self.SolsArray_Full.G[0:ind.size,:,:,:,0,0]=self.SolsArray_G[0:ind.size,:,:,:,0,0]
            self.SolsArray_Full.G[0:ind.size,:,:,:,1,1]=self.SolsArray_G[0:ind.size,:,:,:,1,0]
        else:                
            self.SolsArray_Full.G[0:ind.size]=self.SolsArray_G[0:ind.size]



        # if SaveStats:
        #     ListStd=[l for l in self.ListStd if len(l)>0]
        #     Std=np.array(ListStd)
        #     ListMax=[l for l in self.ListMax if len(l)>0]
        #     Max=np.array(ListMax)
            
        #     ListKapa=[l for l in self.ListKeepKapa if len(l)>0]
        #     Kapa=np.array(ListKapa)
        #     nf,na,nt=Std.shape
        #     NoiseInfo=np.zeros((nf,na,nt,3))
        #     NoiseInfo[:,:,:,0]=Std[:,:,:]
        #     NoiseInfo[:,:,:,1]=np.abs(Max[:,:,:])
        #     NoiseInfo[:,:,:,2]=Kapa[:,:,:]
            
        #     StatFile="NoiseInfo.npy"
        #     log.print( "Saving statistics in %s"%StatFile)
        #     np.save(StatFile,NoiseInfo)


        Sols=self.SolsArray_Full[0:ind.size].copy()
        
        if Sols.size==0:
            na=self.VS.MS.na
            nd=self.SM.NDir
            nChan=self.VS.NChanJones
            Sols=np.zeros((1,),dtype=[("t0",np.float64),
                                      ("t1",np.float64),
                                      ("G",np.complex64,(nChan,na,nd,2,2)),
                                      ("Stats",np.float32,(nChan,na,4))])
            Sols=Sols.view(np.recarray)
            Sols.t0[0]=0
            Sols.t1[0]=self.VS.MS.times_all[-1]
            Sols.G[0,:,:,:,0,0]=1
            Sols.G[0,:,:,:,1,1]=1
            

        Sols=Sols.view(np.recarray)
        Sols.t1[-1]+=1e3
        Sols.t0[0]-=1e3

        return Sols

    def InitSol(self,G=None,TestMode=True):
        na=self.VS.MS.na
        nd=self.SM.NDir
        nChan=self.VS.NChanJones

        if type(G)==type(None):
            if self.PolMode=="Scalar":
                G=np.ones((nChan,na,nd,1,1),self.DType)
            elif self.PolMode=="IDiag":
                G=np.ones((nChan,na,nd,2,1),self.DType)
            else:
                G=np.zeros((nChan,na,nd,2,2),self.DType)
                G[:,:,:,0,0]=1
                G[:,:,:,1,1]=1
            self.HasFirstGuessed=False

        else:
            self.HasFirstGuessed=True
        self.G=G
        #self.G*=0.001
        _,_,_,npolx,npoly=self.G.shape


        # # print "int!!!!!!!!!!"
        # self.G+=np.random.randn(*self.G.shape)*1.#sigP
        
        NSols=np.max([1,int(1.5*round(self.VS.MS.DTh/(self.VS.TVisSizeMin/60.)))])
        #print "Nsols",NSols,self.VS.MS.DTh,self.VS.TVisSizeMin/60.
        

        self.SolsArray_t0=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_t1=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_tm=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_done=np.zeros((NSols,),dtype=np.bool8)
        self.SolsArray_G=np.zeros((NSols,nChan,na,nd,npolx,npoly),dtype=np.complex64)
        self.SolsArray_Stats=np.zeros((NSols,nChan,na,4),dtype=np.float32)

        self.Power0=np.zeros((nChan,na),np.float32)
        self.SolsArray_t0=NpShared.ToShared("%sSolsArray_t0"%self.IdSharedMem,self.SolsArray_t0)
        self.SolsArray_t1=NpShared.ToShared("%sSolsArray_t1"%self.IdSharedMem,self.SolsArray_t1)
        self.SolsArray_tm=NpShared.ToShared("%sSolsArray_tm"%self.IdSharedMem,self.SolsArray_tm)
        self.SolsArray_done=NpShared.ToShared("%sSolsArray_done"%self.IdSharedMem,self.SolsArray_done)
        self.SolsArray_G=NpShared.ToShared("%sSolsArray_G"%self.IdSharedMem,self.SolsArray_G)
        self.SolsArray_Full=np.zeros((NSols,),dtype=[("t0",np.float64),
                                                     ("t1",np.float64),
                                                     ("G",np.complex64,(nChan,na,nd,2,2)),
                                                     ("Stats",np.float32,(nChan,na,4))])
        self.SolsArray_Full=self.SolsArray_Full.view(np.recarray)

        self.DicoKapa={}
        self.DicoKeepKapa={}
        self.DicoStd={}
        self.DicoMax={}
        for (iAnt,iChanSol) in ItP(range(na),range(nChan)):
            self.DicoKapa[(iAnt,iChanSol)]=[]
            self.DicoKeepKapa[(iAnt,iChanSol)]=[]
            self.DicoStd[(iAnt,iChanSol)]=[]
            self.DicoMax[(iAnt,iChanSol)]=[]


        self.G=NpShared.ToShared("%sSharedGains"%self.IdSharedMem,self.G)
        self.G0Iter=NpShared.ToShared("%sSharedGains0Iter"%self.IdSharedMem,self.G.copy())
        #self.InitCovariance()




        
    def InitCovariance(self,FromG=False,sigP=0.1,sigQ=0.01):
        if self.SolverType!="KAFCA": return
        if self.Q!=None: return
        
        na=self.VS.MS.na
        nd=self.SM.NDir
        nChan=self.VS.NChanJones

        
        _,_,_,npol,_=self.G.shape
        


        npolx,npoly=self.NJacobBlocks_X,self.NJacobBlocks_Y

        if FromG==False:
            P=(sigP**2)*np.array([np.diag(np.ones((nd*npolx*npoly,),self.DType)) for iAnt in range(na)])
            Q=(sigQ**2)*np.array([np.diag(np.ones((nd*npolx*npoly,),self.DType)) for iAnt in range(na)])
        else:

            P=(sigP**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*np.diag(np.ones((nd*npolx*npoly),self.DType)) for iAnt in range(na)])
            Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*np.diag(np.ones((nd*npolx*npoly),self.DType)) for iAnt in range(na)])

        if self.SM.ApparentSumI is None:
            self.InitMeanBeam()


        self.VS.giveDataSizeAntenna()
        QList=[]
        PList=[]
        for iChanSol in range(self.VS.NChanJones):
            ra=self.SM.ClusterCat.ra
            dec=self.SM.ClusterCat.dec
            ns=ra.size
            
            d=np.sqrt((ra.reshape((ns,1))-ra.reshape((1,ns)))**2+(dec.reshape((ns,1))-dec.reshape((1,ns)))**2)
            d0=1.*np.pi/180
            QQ=(1./(1.+d/d0))**2
            Qa=np.zeros((nd,npolx,npoly,nd,npolx,npoly),self.DType)
            for ipol in range(npolx):
                for jpol in range(npoly):
                    Qa[:,ipol,jpol,:,ipol,jpol]=QQ[:,:]

            #Qa=np.zeros((nd,npolx,npoly,nd,npolx,npoly),self.DType)
            F=self.SM.ClusterCat.SumI.copy()
            F/=F.max()

            #stop
            #self.SM.ApparentSumI=np.zeros((nd,),np.float32)
            Qa.fill(0)
            ApFluxes=self.NormFluxes*self.AbsMeanBeamAnt**2
            for idir in range(nd):
                #Qa[idir,:,:,idir,:,:]*=(self.SM.ApparentSumI[idir])**2
                #Qa[idir,:,:,idir,:,:]*=ApFluxes[idir]**2
                #Qa[idir,:,:,idir,:,:]=1
                Qa[idir,:,:,idir,:,:]=ApFluxes[idir]


            # for idir in range(nd):
            #     for jdir in range(nd):
            #         Qa[idir,:,:,jdir,:,:]=np.sqrt(ApFluxes[idir]*ApFluxes[jdir])*QQ[idir,jdir]

            # import pylab
            # pylab.clf()
            # pylab.imshow(QQ,interpolation="nearest")
            # pylab.draw()
            # pylab.show()

            
            Qa=Qa.reshape((nd*npolx*npoly,nd*npolx*npoly))
            #print np.diag(Qa)
            #Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iChanSol,iAnt]))**2*(Qa*(self.VS.fracNVisPerAnt[iAnt]**4))**(self.GD["KAFCA"]["PowerSmooth"]) for iAnt in range(na)])
            Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iChanSol,iAnt]))**2*Qa for iAnt in range(na)])
            #Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iChanSol,iAnt]))**2*(Qa*(self.VS.Compactness[iAnt]**2*self.VS.fracNVisPerAnt[iAnt]**4))**(self.GD["KAFCA"]["PowerSmooth"]) for iAnt in range(na)])
            #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            #Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iChanSol,iAnt]))**2*Qa for iAnt in range(na)])
            #print Q[0]

            QList.append(Q)
            PList.append(P)
        Q=np.array(QList)
        P=np.array(PList)
        
        self.P=P
        self.evP=np.zeros_like(P)
        self.P=NpShared.ToShared("%sSharedCovariance"%self.IdSharedMem,self.P)
        self.Q=NpShared.ToShared("%sSharedCovariance_Q"%self.IdSharedMem,Q)
        self.Q_Init=self.Q.copy()

        self.evP=NpShared.ToShared("%sSharedEvolveCovariance"%self.IdSharedMem,self.evP)
        nbuff=10

    def InitMeanBeam(self):

        self.NormFluxes=self.SM.ClusterCat.SumI.copy()
        # print np.sort(self.NormFluxes)
        # FCut=5.
        # self.NormFluxes[self.NormFluxes>FCut]=FCut
        
        self.NormFluxes/=self.NormFluxes.max()
        if self.GD["Beam"]["BeamModel"] is None:
            self.SM.ApparentSumI=self.NormFluxes
            self.SM.AbsMeanBeamAnt=np.ones_like(self.SM.ApparentSumI)
            self.AbsMeanBeamAnt=self.SM.AbsMeanBeamAnt
        else:
            nd=self.SM.ClusterCat.SumI.size
            self.SM.ApparentSumI=np.zeros((nd,),np.float32)
            from killMS.Data import ClassBeam
            log.print( "Calculate mean beam for covariance estimate... ")
            BeamMachine=ClassBeam.ClassBeam(self.VS.MSName,self.GD,self.SM)
            AbsMeanBeam=BeamMachine.GiveMeanBeam()
            AbsMeanBeamAnt=np.mean(AbsMeanBeam[:,:,0,0,0],axis=1)

            self.AbsMeanBeamAnt=AbsMeanBeamAnt
            self.SM.ApparentSumI=(AbsMeanBeamAnt)*self.NormFluxes
            self.SM.AbsMeanBeamAnt=AbsMeanBeamAnt

        # pylab.clf()
        # pylab.scatter(self.SM.ClusterCat.l,self.SM.ClusterCat.m,c=self.SM.ApparentSumI)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop

        self.InitReg()

    def InitReg(self):
        if self.SolverType=="KAFCA": return
        if self.GD["CohJones"]["LambdaTk"]==0: return
        NDir=self.SM.NDir
        X0=np.ones((NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y),dtype=np.float32)
        L=np.ones((NDir,self.NJacobBlocks_X,self.NJacobBlocks_Y),dtype=np.float32)
        if self.PolMode=="IFull":
            X0[:,0,1]=0
            X0[:,1,0]=0
                
        SumI=self.SM.ClusterCat.SumI.copy()
        SumIApp=SumI*self.SM.AbsMeanBeamAnt**2

        MaxFlux=1.
        #indFree=np.where(SumIApp>MaxFlux)[0]
        #SumIApp[indFree]=MaxFlux

        #SumIAppNorm=SumIApp#/MaxFlux
        #Linv=L/SumIAppNorm.reshape((NDir,1,1))

        AbsMeanBeamAntsq=self.SM.AbsMeanBeamAnt**2
        Linv=L/AbsMeanBeamAntsq.reshape((NDir,1,1))


        Linv=Linv**2

        log.print( "Using Tikhonov regularisation [LambdaTk = %.2f]"%self.GD["CohJones"]["LambdaTk"])
        #log.print( "  there are %i free directions"%indFree.size)
        log.print( "  minimum inverse L-matrix is %.3f"%Linv.min())
        log.print( "  maximum inverse L-matrix is %.3f"%Linv.max())
        # for iDir in range(NDir):
        #     log.print( "  #%i : [%7.3fJy x %7.7f] %7.3f Jy -> %7.7f "%(iDir,SumI[iDir],self.SM.AbsMeanBeamAnt[iDir],SumIApp[iDir],Linv.flat[iDir]))

        NpShared.ToShared("%sLinv"%self.IdSharedMem,Linv)
        NpShared.ToShared("%sX0"%self.IdSharedMem,X0)
        
    def setNextData(self):
        DATA=self.VS.GiveNextVis()
        if self.VS_PredictCol is not None:
            self.VS_PredictCol.GiveNextVis()

        NDone,nt=self.pBarProgress
        intPercent=int(100*  NDone / float(nt))
        self.pBAR.render(NDone,nt)

        if DATA=="EndOfObservation":
            log.print( ModColor.Str("Reached end of data"))
            return "EndOfObservation"
        if DATA=="EndChunk":
            log.print( ModColor.Str("Reached end of data chunk"))
            return "EndChunk"
        if DATA=="AllFlaggedThisTime":
            #log.print( ModColor.Str("AllFlaggedThisTime"))
            self.AppendGToSolArray()
            self.iCurrentSol+=1
            return "AllFlaggedThisTime"

        
        ## simul
        #d=self.DATA["data"]
        #self.DATA["data"]+=(self.rms/np.sqrt(2.))*(np.random.randn(*d.shape)+1j*np.random.randn(*d.shape))
        self.DATA=DATA

        self.rms=-1
        if (self.TypeRMS=="Resid")&(self.rmsFromData!=None):
            self.rms=self.rmsFromData
            #log.print(" rmsFromDataJacobAnt: %s"%self.rms)
        elif self.rmsFromExt!=None:
            self.rms=self.rmsFromExt
            #log.print(" rmsFromExt: %s"%self.rms)
        elif (self.TypeRMS=="GlobalData"):
            nrow,nch,_=DATA["flags"].shape
            if self.VS.MS.NPolOrig==4:
                Dpol=DATA["data"][:,:,1:3]
                Fpol=DATA["flags"][:,:,1:3]
                w=DATA["W"].reshape((nrow,nch,1))*np.ones((1,1,2))
            else:
                Dpol=DATA["data"][:,:,0:1]
                Fpol=DATA["flags"][:,:,0:1]
                w=DATA["W"].reshape((nrow,nch,1))
                
            self.rms=np.sqrt(np.sum((w[Fpol==0]*np.absolute(Dpol[Fpol==0]))**2.0)/np.sum(w[Fpol==0]**2.0))/np.sqrt(2.)
            # print
            # log.print(" rmsFromGlobalData: %s"%self.rms)
            # print DATA["data"].shape
            # print
        else:
            stop

        
        #print("rms=",self.rms)

        return True

    def SetRmsFromExt(self,rms):
        self.rmsFromExt=rms

    def InitPlotGraph(self):
        from Plot import Graph
        log.print("Initialising plots ..." )
        import pylab
        #pylab.ion()
        self.Graph=Graph.ClassMplWidget(self.VS.MS.na)
        
        for iAnt in range(self.VS.MS.na):
            self.Graph.subplot(iAnt)
            self.Graph.imshow(np.zeros((10,10),dtype=np.float32),interpolation="nearest",aspect="auto",origin='lower',vmin=0.,vmax=2.)#,extent=(-3,3,-3,3))
            self.Graph.text(0,0,self.VS.MS.StationNames[iAnt])
            self.Graph.draw()

        pylab.draw()
        pylab.show(False)


        

    #################################
    ##          Serial             ## 
    #################################

    def doNextTimeSolve(self,SkipMode=False):

        import pylab

        if type(self.G)==type(None):
            self.InitSol()

        ListAntSolve=[i for i in range(self.VS.MS.na) if not(i in self.VS.FlagAntNumber)]
        self.DicoJM={}

        self.pBAR= ProgressBar(Title="Solving ")
        if not(self.DoPBar): self.pBAR.disable()
        NDone=0
        T0,T1=self.VS.TimeMemChunkRange_sec[0],self.VS.TimeMemChunkRange_sec[1]
        DT=(T1-T0)
        dt=self.VS.TVisSizeMin*60.
        nt=int(DT/float(dt))+1
        #pBAR.disable()
        self.pBAR.render(0, '%4i/%i' % (0,nt))

        T=ClassTimeIt.ClassTimeIt("WirtingerSolver")
        T.disable()

        iiCount=0
        while True:
            self.pBarProgress=NDone,float(nt)
            NDone+=1
            T.reinit()

            #print
            #print "zeros=",np.count_nonzero(NpShared.GiveArray("%sPredictedData"%self.IdSharedMem))
            #print
            Res=self.setNextData()
            if Res=="EndChunk": break
            T.timeit("read data")

            if SkipMode:
                print(iiCount)
                iiCount+=1
                if iiCount<585: continue
            
            t0,t1=self.VS.CurrentVisTimes_MS_Sec
            self.SolsArray_t0[self.iCurrentSol]=t0
            self.SolsArray_t1[self.iCurrentSol]=t1
            tm=(t0+t1)/2.
            self.SolsArray_tm[self.iCurrentSol]=tm
            ThisTime=tm
            T.timeit("stuff")
            for (iAnt,iChanSol) in ItP(ListAntSolve,range(self.VS.NChanJones)):
                #print iAnt,iChanSol
                ch0,ch1=self.VS.SolsToVisChanMapping[iChanSol]
                SharedDicoDescriptors={"SharedVis":self.VS.SharedVis_Descriptor,
                                       "PreApplyJones":self.VS.PreApplyJones_Descriptor,
                                       "SharedAntennaVis":None,
                                       "DicoClusterDirs":self.VS.DicoClusterDirs_Descriptor}

                

                JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Precision="S",IdSharedMem=self.IdSharedMem,GD=self.GD,
                                        ChanSel=(ch0,ch1),
                                        SharedDicoDescriptors=SharedDicoDescriptors,
                                        **self.ConfigJacobianAntenna)
                    
                T.timeit("JM")
                JM.setDATA_Shared()
                T.timeit("Setdata_Shared")
                self.DicoJM[(iAnt,iChanSol)]=JM

            T.timeit("Class")
            
            if (self.CounterEvolveP())&(self.SolverType=="KAFCA")&(self.iCurrentSol>self.EvolvePStepStart):
                print("Evolve0")
                for (iAnt,iChanSol) in ItP(ListAntSolve,range(self.VS.NChanJones)):
                    JM=self.DicoJM[(iAnt,iChanSol)]
                    self.evP[iChanSol,iAnt]=JM.CalcMatrixEvolveCov(self.G[iChanSol],self.P[iChanSol],self.rms)

            elif (self.SolverType=="KAFCA")&(self.iCurrentSol<=self.EvolvePStepStart):
                print("Evolve1")
                for (iAnt,iChanSol) in ItP(ListAntSolve,range(self.VS.NChanJones)):
                    JM=self.DicoJM[(iAnt,iChanSol)]
                    self.evP[iChanSol,iAnt]=JM.CalcMatrixEvolveCov(self.G[iChanSol],self.P[iChanSol],self.rms)


            T.timeit("Evolve")
            #print
            Dico_SharedDicoDescriptors={}
            for iChanSol in range(self.VS.NChanJones):
                #print
                # Reset Data
                NpShared.DelAll("%sDicoData"%self.IdSharedMem)

                for LMIter in range(self.NIter):
                    Gnew=self.G.copy()
                    if self.SolverType=="KAFCA":
                        Pnew=self.P.copy()
                    for iAnt in ListAntSolve:
                        JM=self.DicoJM[(iAnt,iChanSol)]
                        if LMIter!=0:
                            JM.SharedDicoDescriptors["SharedAntennaVis"]=Dico_SharedDicoDescriptors[iAnt]
                        
                        if self.SolverType=="CohJones":
                            
                            x,_,InfoNoise=JM.doLMStep(self.G[iChanSol])
                            T.timeit("LMStep")
                            if LMIter==self.NIter-1: 
                                print("!!!!!!!!!!!!!!!!!!!")
                                # self.G.fill(1)
                                JM.PredictOrigFormat(self.G[iChanSol])
                                T.timeit("PredictOrig")
                                
                        if self.SolverType=="KAFCA":

                            EM=ClassModelEvolution(iAnt,iChanSol,
                                                   StepStart=3,
                                                   WeigthScale=0.3,
                                                   DoEvolve=True,
                                                   BufferNPoints=10,
                                                   sigQ=0.01,IdSharedMem=self.IdSharedMem)

                            # print self.G[iChanSol]
                            # if iAnt==3 and NDone==2:
                            #     print self.G[iChanSol]
                            #     stop

                            #print self.G.ravel()[0::5],self.Q.ravel()[0::5],self.evP.ravel()[0::5],self.P.ravel()[0::5]
                            print(self.evP.ravel()[0::5])
                            if NDone==1: stop
                            x,P,InfoNoise=JM.doEKFStep(self.G[iChanSol],self.P[iChanSol],self.evP[iChanSol],self.rms)
                            if LMIter==self.NIter-1: JM.PredictOrigFormat(self.G[iChanSol])

                            
                            xe=None
    
                            Pa=EM.Evolve0(x,P)

    
                            #xe,Pa=EM.Evolve(x,P,ThisTime)
                            if Pa!=None:
                                P=Pa
                            if xe!=None:
                                x=xe
    
    
                            Pnew[iChanSol,iAnt]=P
                            self.P[iChanSol,iAnt,:]=P[:]
                            # if NDone==2:
                            #     print iAnt,x#,P#,self.G[iChanSol],Pa
                            # if NDone==3:
                            #     stop
                            # print iAnt,np.sum(Pnew[iChanSol,iAnt])

                        Dico_SharedDicoDescriptors[iAnt]=JM.SharedDicoDescriptors["SharedAntennaVis"]
                        Gnew[iChanSol,iAnt]=x
                        T.timeit("  SolveAnt %i [%i->%i]"%(iAnt,JM.ch0,JM.ch1))
                
                        kapa=InfoNoise["kapa"]
                        self.DicoStd[iAnt,iChanSol].append(InfoNoise["std"])
                        self.DicoMax[iAnt,iChanSol].append(InfoNoise["max"])
                        self.DicoKeepKapa[iAnt,iChanSol].append(InfoNoise["kapa"])
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][0]=InfoNoise["std"]
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][1]=InfoNoise["max"]
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][2]=InfoNoise["kapa"]
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][3]=self.rms

                        if (kapa!=None)&(LMIter==0):
                            if kapa==-1.:
                                if len(self.DicoKapa[iAnt,iChanSol])>0:
                                    kapa=self.DicoKapa[iAnt,iChanSol][-1]
                                else:
                                    kapa=1.
    
                            self.DicoKapa[iAnt,iChanSol].append(kapa)
                            dt=.5
                            TraceResidList=self.DicoKapa[iAnt,iChanSol]
                            x=np.arange(len(TraceResidList))
                            expW=np.exp(-x/dt)[::-1]
                            expW/=np.sum(expW)
                            kapaW=np.sum(expW*np.array(TraceResidList))
                            self.Q[iChanSol,iAnt][:]=(kapaW)*self.Q_Init[iChanSol,iAnt][:]
                            
                    # pylab.figure(1)
                    # pylab.clf()
                    # pylab.plot(np.abs(Gnew[iChanSol].flatten()))
                    # pylab.plot(np.abs(self.G[iChanSol].flatten()))
                    # pylab.title("Channel=%i"%iChanSol)
                    # pylab.ylim(0,2)
                    # pylab.draw()
                    # pylab.show(False)
                    # pylab.pause(0.1)

                    self.G[iChanSol]=Gnew[iChanSol]


                # pylab.figure(1)
                # pylab.clf()
                # pylab.plot(np.abs(Gnew[iChanSol].flatten()))
                # if self.SolverType=="KAFCA":
                #     sig=np.sqrt(np.array([np.diag(Pnew[iChanSol,iAnt]) for iAnt in range(self.VS.MS.na)]).flatten())
                #     pylab.plot(np.abs(Gnew[iChanSol].flatten())+sig,color="black",ls="--")
                #     pylab.plot(np.abs(Gnew[iChanSol].flatten())-sig,color="black",ls="--")
                #     self.P[:]=Pnew[:]
                # pylab.plot(np.abs(self.G[iChanSol].flatten()))
                # pylab.ylim(0,2)
                # pylab.draw()
                # pylab.show(False)
                # self.G[:]=Gnew[:]

                JM.SharedDicoDescriptors["SharedAntennaVis"]=None
                T.timeit("Plot")

            #print self.P.ravel()
            #if NDone==1: stop

            self.AppendGToSolArray()
            #self.SolsArray_done[self.iCurrentSol]=1
            #self.SolsArray_G[self.iCurrentSol][:]=self.G[:]
            
            self.iCurrentSol+=1

        return True



    # #################################
    # ###        Parallel           ###
    # #################################
    
    
    
    def doNextTimeSolve_Parallel(self,OnlyOne=False,SkipMode=False,Parallel=True):

        
        
        Parallel=True
        #Parallel=False
        #SkipMode=True
        
        ListAntSolve=[i for i in range(self.VS.MS.na) if not(i in self.VS.FlagAntNumber)]

        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()




        workerlist=[]
        NCPU=self.NCPU

        import time
        
                    


        #T=ClassTimeIt.ClassTimeIt()
        #T.disable()

        JonesToVisChanMapping=self.VS.SolsToVisChanMapping
        for ii in range(NCPU):
             
            W=WorkerAntennaLM(work_queue, result_queue,self.SM,self.PolMode,self.SolverType,self.IdSharedMem,
                              ConfigJacobianAntenna=self.ConfigJacobianAntenna,GD=self.GD,SM_Compress=self.SM_Compress,
                              JonesToVisChanMapping=JonesToVisChanMapping)#,args=(e,))
            workerlist.append(W)
            if Parallel:
                workerlist[ii].start()


        ##############################

        T0,T1=self.VS.TimeMemChunkRange_sec[0],self.VS.TimeMemChunkRange_sec[1]
        DT=(T1-T0)
        dt=self.VS.TVisSizeMin*60.
        dt=np.min([dt,DT])
        nt=int(np.ceil(DT/float(dt)))
        # if DT/float(dt)-nt>1.:
        #     nt+=1
        #nt=np.max([1,nt])
        
        log.print("DT=%f, dt=%f, nt=%f"%(DT,dt,nt))
        

        self.pBAR= ProgressBar(Title="Solving ")
        if not(self.DoPBar): self.pBAR.disable()
        

        
        self.pBAR.render(0,nt)
        NDone=0
        iiCount=0
        ThisG=self.G.copy()
        if self.SolverType=="KAFCA":
            ThisP=self.P.copy()
            ThisQ=self.Q.copy()
            
        while True:
            T=ClassTimeIt.ClassTimeIt("ClassWirtinger DATA[%4.4i]"%NDone)
            T.disable()
            T.reinit()
            self.pBarProgress=NDone,float(nt)
            Res=self.setNextData()
            NDone+=1
            T.timeit("read data")
            if Res=="EndChunk":
                break
            if Res=="AllFlaggedThisTime":
                continue
            #print "saving"
            #print "saving"
            #sols=self.GiveSols()
            #np.save("lastSols",sols)
            #print "done"
            if SkipMode:

                #continue
                iiCount+=1
                if iiCount<=869:
                    print(iiCount)
                    print(iiCount)
                    continue

            # iiCount+=1
            # print("iiCount",self.VS.CurrentMemTimeChunk,iiCount)
            # # if self.VS.CurrentMemTimeChunk!=4: continue
            # # if iiCount<7: continue
            # print("   doSoleve")
            

            t0,t1=self.VS.CurrentVisTimes_MS_Sec
            tm=(t0+t1)/2.
  
            NJobs=len(ListAntSolve)
            NTotJobs=NJobs*self.NIter

            lold=0
            iResult=0

            T.timeit("stuff")

            if (not(self.HasFirstGuessed))&(self.SolverType=="CohJones"):
                NIter=15
                self.HasFirstGuessed=True
            else:
                NIter=self.NIter


            #print "!!!!!!!!!!!!!!!!!!!!!!!!!"
            #NIter=1


            Gold=self.G.copy()
            DoCalcEvP=np.zeros((self.VS.NChanJones,),bool)            
            DoCalcEvP[:]=False
            if (self.CounterEvolveP())&(self.SolverType=="KAFCA")&(self.iCurrentSol>self.EvolvePStepStart):
                DoCalcEvP[:]=True
            elif (self.SolverType=="KAFCA")&(self.iCurrentSol<=self.EvolvePStepStart):
                DoCalcEvP[:]=True

            T.timeit("before iterloop")
            u,v,w=self.DATA["uvw"].T
            A0=self.DATA["A0"]
            A1=self.DATA["A1"]
            meanW=np.zeros((self.VS.MS.na,),np.float32)
            for iAntMS in ListAntSolve:
                ind=np.where((A0==iAntMS)|(A1==iAntMS))[0]
                if ind.size==0: continue
                meanW[iAntMS]=np.mean(np.abs(w[ind]))
            meanW=meanW[ListAntSolve]
            indOrderW=np.argsort(meanW)[::-1]
            SortedWListAntSolve=(np.array(ListAntSolve)[indOrderW]).tolist()
            #print indOrderW
            #NpShared.ToShared("%sSharedGainsPrevious"%self.IdSharedMem,self.G.copy())
            #NpShared.ToShared("%sSharedPPrevious"%self.IdSharedMem,self.P.copy())
            Dico_SharedDicoDescriptors={}

            if self.SolPredictMachine is not None:
                t0_ms,t1_ms=self.VS.CurrentVisTimes_MS_Sec
                tm_ms=(t0_ms+t1_ms)/2.
                xPredict=self.SolPredictMachine.GiveClosestSol(tm_ms,
                                                               self.VS.SolsFreqDomains,np.arange(self.VS.MS.na),
                                                               self.SM.ClusterCat.ra,self.SM.ClusterCat.dec)
                
                # nChan,na,nd,2,2
                if self.PolMode=="Scalar":
                    self.G[:,:,:,0,0]=xPredict[:,:,:,0,0]
                elif self.PolMode=="IDiag":
                    self.G[:,:,:,0,0]=xPredict[:,:,:,0,0]
                    self.G[:,:,:,1,0]=xPredict[:,:,:,1,1]
                else:
                    self.G[:]=xPredict[:]

            DoEvP=np.zeros((self.VS.NChanJones,),bool)
            for iChanSol in range(self.VS.NChanJones):
                # # Reset Data
                # # _,na,_,_,_=self.G.shape
                # g=self.G[iChanSol,:,:,0,0]
                # P=np.mean(np.abs(g)**2,1)
                # if self.iCurrentSol!=0:
                #     fact=self.Power0[iChanSol]/P
                #     self.G[iChanSol]*=fact.reshape((na,1,1,1))
                #     print fact,self.Power0[iChanSol],P
                # else:
                #     self.Power0[iChanSol]=P
                #     print self.Power0[iChanSol]

                NpShared.DelAll("%sDicoData"%self.IdSharedMem)
                for LMIter in range(NIter):

                    ThisG[iChanSol,:]=self.G[iChanSol,:]

                    if self.SolverType=="KAFCA":
                        ThisP[iChanSol,:]=self.P[iChanSol,:]
                        ThisQ[iChanSol,:]=self.Q[iChanSol,:]

                    #print
                    # for EKF
    
                    #print "===================================================="
                    #print "===================================================="
                    #print "===================================================="
                    #########
                    if LMIter>0:
                        DoCalcEvP[iChanSol]=False
                        DoEvP[iChanSol]=False
                    elif LMIter==0:
                        self.G0Iter[iChanSol,:]=ThisG[iChanSol,:]
                        DoEvP[iChanSol]=False
    
                    if LMIter==(NIter-1):
                        DoEvP[iChanSol]=True
                    
                    DoFullPredict=False
                    if LMIter==(NIter-1):
                        DoFullPredict=True
                        
                    #print self.G.ravel()[0::5],self.Q.ravel()[0::5],self.evP.ravel()[0::5],self.P.ravel()[0::5]
                    #print self.evP.ravel()#[0::5]
                    #if NDone==2: stop

                    #print LMIter,NIter,DoFullPredict
                    for iAnt in SortedWListAntSolve:
                        SharedDicoDescriptors={"SharedVis":self.VS.SharedVis_Descriptor,
                                               "PreApplyJones":self.VS.PreApplyJones_Descriptor,
                                               "SharedAntennaVis":None,
                                               "DicoClusterDirs":self.VS.DicoClusterDirs_Descriptor}
                        if LMIter!=0:
                            SharedDicoDescriptors["SharedAntennaVis"]=Dico_SharedDicoDescriptors[iAnt]
                        # print("iAnt",iAnt)
                        # print(SharedDicoDescriptors)
                        work_queue.put((iAnt,iChanSol,
                                        DoCalcEvP[iChanSol],tm,
                                        self.rms,DoEvP[iChanSol],
                                        DoFullPredict,
                                        SharedDicoDescriptors))
                        # work_queue.put((iAnt,iChanSol,DoCalcEvP[iChanSol],tm,self.rms,DoEvP[iChanSol],
                        #                 DoFullPredict))
     
                    if not Parallel:
                        for ii in range(NCPU):
                            workerlist[ii].run()

                    T.timeit("put in queue")
                    rmsFromDataList=[]
                    DTs=np.zeros((self.VS.MS.na,),np.float32)
                    while iResult < NJobs:
                        iAnt,iChanSol,G,P,rmsFromData,InfoNoise,DT,SharedDicoDescriptors = result_queue.get()
                        Dico_SharedDicoDescriptors[iAnt]=SharedDicoDescriptors
                        if rmsFromData!=None:
                            rmsFromDataList.append(rmsFromData)
                        
                        T.timeit("[%i,%i] get"%(LMIter,iAnt))
                        #"TIMING DIFFERS BETWEEN SINGLE AND PARALLEL_NCPU=1"
                        #stop
                        #T.timeit("result_queue.get()")
                        ThisG[iChanSol,iAnt][:]=G[:]
                        #self.G[iChanSol,iAnt][:]=G[:]
                        if type(P)!=type(None):
                            #P.fill(0.1)
                            ThisP[iChanSol,iAnt,:]=P[:]

                        DTs[iAnt]=DT
                        kapa=InfoNoise["kapa"]
                        self.DicoStd[iAnt,iChanSol].append(InfoNoise["std"])
                        self.DicoMax[iAnt,iChanSol].append(InfoNoise["max"])
                        self.DicoKeepKapa[iAnt,iChanSol].append(InfoNoise["kapa"])
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][0]=InfoNoise["std"]
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][1]=InfoNoise["max"]
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][2]=InfoNoise["kapa"]
                        self.SolsArray_Stats[self.iCurrentSol][iChanSol,iAnt][3]=self.rms
                        
                        # if iAnt==1 and NDone==2:
                        #     print G.ravel(),P.ravel(),
                        #     stop

                        iResult+=1
                        if (kapa!=None)&(LMIter==0):
                            if kapa==-1.:
                                if len(self.DicoKapa[iAnt,iChanSol])>0:
                                    kapa=self.DicoKapa[iAnt,iChanSol][-1]
                                else:
                                    kapa=1.
    
                            self.DicoKapa[iAnt,iChanSol].append(kapa)
                            dt=.5
                            TraceResidList=self.DicoKapa[iAnt,iChanSol]
                            x=np.arange(len(TraceResidList))
                            expW=np.exp(-x/dt)[::-1]
                            expW/=np.sum(expW)
                            kapaW=np.sum(expW*np.array(TraceResidList))
                            #self.Q[iAnt]=(kapaW**2)*self.Q_Init[iAnt]
                            #print kapaW
                            ThisQ[iChanSol,iAnt][:]=(kapaW)*self.Q_Init[iChanSol,iAnt][:]
                            #print("kapa",kapaW)
                            # print self.Q[iChanSol,iAnt]
                            # print self.Q_Init[iChanSol,iAnt]
                            # print

                            # self.Q[iAnt][:]=(kapaW)**2*self.Q_Init[iAnt][:]*1e6
                            # QQ=NpShared.FromShared("%sSharedCovariance_Q"%self.IdSharedMem)[iAnt]
                            # print self.Q[iAnt]-QQ[iAnt]
                            
                            #self.Q[iAnt]=self.Q_Init[iAnt]
    
                            #print iAnt,kapa,kapaW
                            #sig=np.sqrt(np.abs(np.array([np.diag(self.P[i]) for i in [iAnt]]))).flatten()
                            #print sig
                        

                    T.timeit("[%i] OneIter"%LMIter)
                    if len(rmsFromDataList)>0:
                        self.rmsFromData=np.min(rmsFromDataList)
                    iResult=0
    
                    # pylab.clf()
                    # pylab.subplot(2,1,1)
                    # pylab.plot(DTs)
                    # pylab.subplot(2,1,2)
                    # pylab.plot(meanW)
                    # pylab.draw()
                    # pylab.show(False)
                    # pylab.pause(0.1)
    
                    if self.DoPlot==1:
                        import pylab
                        pylab.figure(1)
                        AntPlot=np.arange(self.VS.MS.na)#np.array(ListAntSolve)
                        pylab.clf()
                        pylab.plot(np.abs(ThisG[iChanSol,AntPlot].flatten()))
                        pylab.plot(np.abs(Gold[iChanSol,AntPlot].flatten()))
                        
                        if self.SolverType=="KAFCA":
    
                            sig=[]
                            for iiAnt in AntPlot:
                                xx=np.array([np.diag(ThisP[iChanSol,iiAnt]) ])
                                if iiAnt in ListAntSolve:
                                    sig.append(np.sqrt(np.abs(xx)).flatten().tolist())
                                else:
                                    sig.append(np.zeros((xx.size,),ThisP.dtype).tolist())
                            
                            sig=np.array(sig).flatten()
    
                            pylab.plot(np.abs(ThisG[iChanSol,AntPlot].flatten())+sig,color="black",ls="--")
                            pylab.plot(np.abs(ThisG[iChanSol,AntPlot].flatten())-sig,color="black",ls="--")
                        pylab.title("Channel=%i"%iChanSol)
                        #pylab.ylim(0,2)
                        pylab.draw()
                        pylab.show(block=False)
                        pylab.pause(0.1)
                        
    
                    T.timeit("[%i] Plot"%LMIter)

                    self.G[iChanSol,:]=ThisG[iChanSol,:]
                    if self.SolverType=="KAFCA":
                        self.P[iChanSol,:]=ThisP[iChanSol,:]
                        self.Q[iChanSol,:]=ThisQ[iChanSol,:]
                        nf,na,nd,nd=self.Q.shape
                        P=np.zeros_like(self.P)
                        Q=np.zeros_like(self.Q)
                        # for iChan in range(nf):
                        #     for iAnt in range(na):
                        #         P[iChan,iAnt,:,:]=np.diag(np.diag(self.P[iChan,iAnt,:,:]))
                        #         Q[iChan,iAnt,:,:]=np.diag(np.diag(self.Q[iChan,iAnt,:,:]))
                        for iAnt in range(na):
                            P[iChanSol,iAnt,:,:]=np.diag(np.diag(self.P[iChanSol,iAnt,:,:]))
                            Q[iChanSol,iAnt,:,:]=np.diag(np.diag(self.Q[iChanSol,iAnt,:,:]))
                        self.P[iChanSol,:]=P[iChanSol,:]
                        self.Q[iChanSol,:]=Q[iChanSol,:]
                        

                # self.G[:]=ThisG[:]
                # #self.G[:]=G[:]
                # #print self.G.shape,G.shape
                # if self.SolverType=="KAFCA":
                #     self.P[:]=ThisP[:]
                #     self.Q[:]=ThisQ[:]

                
                # end Niter
            # end Chan



            #print self.P.ravel()
            #if NDone==1: stop
            
            self.AppendGToSolArray()
            T.timeit("AppendGToSolArray")

            self.iCurrentSol+=1


            #_T=ClassTimeIt.ClassTimeIt("Plot")
            #_T.timeit()
            if self.DoPlot==2:
                S=self.GiveSols()
                #print S.G[-1,0,:,0,0]
                for ii in range(S.G.shape[1]):
                    self.Graph.subplot(ii)
                    self.Graph.imshow(np.abs(S.G[:,ii,:,0,0]).T)
                    #self.Graph.imshow(np.random.randn(*(S.G[:,ii,:,0,0]).shape))
                    self.Graph.text(0,0,self.VS.MS.StationNames[ii])
                self.Graph.draw()
                self.Graph.savefig()
            #_T.timeit()

            T.timeit("Ending")
            #if NDone==1:
            #    break
            if OnlyOne: break
        # end while chunk


        # if self.SolverType=="KAFCA":
        #     np.save("P.%s.npy"%self.GD["Solutions"]['OutSolsName'],np.array(self.PListKeep))
        #     np.savez("P.%s.npz"%self.GD["Solutions"]['OutSolsName'],
        #              ListP=np.array(self.PListKeep),
        #              ListQ=np.array(self.QListKeep))

        if Parallel:
            for ii in range(NCPU):
                workerlist[ii].shutdown()
                workerlist[ii].terminate()
                workerlist[ii].join()

#        print self.G.ravel()
#        stop
            
        return True


    def AppendGToSolArray(self):
        t0,t1=self.VS.CurrentVisTimes_MS_Sec
        self.SolsArray_t0[self.iCurrentSol]=t0
        self.SolsArray_t1[self.iCurrentSol]=t1
        tm=(t0+t1)/2.
        self.SolsArray_tm[self.iCurrentSol]=tm
        self.SolsArray_done[self.iCurrentSol]=1
        self.SolsArray_G[self.iCurrentSol][:]=self.G[:]

        # if self.SolverType=="KAFCA":
        #     self.PListKeep.append(self.P.copy())
        #     self.QListKeep.append(self.Q.copy())
        # NDone=np.count_nonzero(self.SolsArray_done)
        # print(NDone)
        # print(NDone)
        # print(NDone)
        # if NDone>=867:
        #     FileName="CurrentSols.%i.npy"%NDone
        #     log.print( "Save Solutions in file: %s"%FileName)
        #     log.print( "Save Solutions in file: %s"%FileName)
        #     log.print( "Save Solutions in file: %s"%FileName)
        #     Sols=self.GiveSols()
            
        #     np.save(FileName,Sols.G.copy())
        #     if self.SolverType=="KAFCA":
        #         # np.save("P.%s.npy"%self.GD["Solutions"]['OutSolsName'],np.array(self.PListKeep))
        #         FName="P.%s.%i.npz"%(self.GD["Solutions"]['OutSolsName'],NDone)
        #         log.print( "Save PQ in file: %s"%FName)
        #         log.print( "Save PQ in file: %s"%FName)
        #         log.print( "Save PQ in file: %s"%FName)
        #         np.savez(FName,
        #                  ListP=np.array(self.PListKeep),
        #                  ListQ=np.array(self.QListKeep))
        
        # if self.SolverType=="KAFCA":
        #     self.PListKeep.append(self.P.copy())
        #     self.QListKeep.append(self.Q.copy())
        #     np.save("P.%s.npy"%self.GD["Solutions"]['OutSolsName'],np.array(self.PListKeep))
        #     np.savez("P.%s.npz"%self.GD["Solutions"]['OutSolsName'],
        #              ListP=np.array(self.PListKeep),
        #              ListQ=np.array(self.QListKeep))




#======================================
import multiprocessing
from killMS.Predict.PredictGaussPoints_NumExpr5 import ClassPredict
class WorkerAntennaLM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,SM,PolMode,SolverType,IdSharedMem,ConfigJacobianAntenna=None,
                 GD=None,JonesToVisChanMapping=None,SM_Compress=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM

        self.SM_Compress=SM_Compress
        self.PolMode=PolMode
        self.SolverType=SolverType
        self.IdSharedMem=IdSharedMem
        self.ConfigJacobianAntenna=ConfigJacobianAntenna
        self.GD=GD
        self.JonesToVisChanMapping=JonesToVisChanMapping

        self.InitPM()

        #self.DoCalcEvP=DoCalcEvP
        #self.ThisTime=ThisTime
        #self.e,=kwargs["args"]
        

    def InitPM(self):

        x=np.linspace(0.,15,100000)
        Exp=np.float32(np.exp(-x))
        LExp=[Exp,x[1]-x[0]]

        Sinc=np.zeros(x.shape,np.float32)
        Sinc[0]=1.
        Sinc[1::]=np.sin(x[1::])/(x[1::])
        LSinc=[Sinc,x[1]-x[0]]

        
        self.PM=ClassPredict(Precision="S",DoSmearing=self.GD["SkyModel"]["Decorrelation"],
                             IdMemShared=self.IdSharedMem,
                             LExp=LExp,LSinc=LSinc,
                             BeamAtFacet=(self.GD["Beam"]["BeamAt"].lower() == "facet"))

        if self.GD["ImageSkyModel"]["BaseImageName"]!="" and self.GD["SkyModel"]["SkyModelCol"] is None:
            self.PM.InitGM(self.SM)

        self.PM_Compress=None
        if self.SM_Compress:
            self.PM_Compress=ClassPredict(Precision="S",
                                          DoSmearing=self.GD["SkyModel"]["Decorrelation"],
                                          IdMemShared=self.IdSharedMem,
                                          LExp=LExp,LSinc=LSinc)
            
    def shutdown(self):
        self.exit.set()
    def run(self):

        ####################
        # Parallel
        while not self.kill_received:# and not self.work_queue.qsize()==0:
            iAnt,iChanSol,DoCalcEvP,ThisTime,rms,DoEvP,DoFullPredict,SharedDicoDescriptors = self.work_queue.get()

            
        # ####################
        # # Serial
        # while not self.kill_received and not self.work_queue.qsize()==0:
        #     #iAnt,iChanSol,DoCalcEvP,ThisTime,rms,DoEvP,DoFullPredict,SharedDicoDescriptors = self.work_queue.get(True,2)
        #     #iAnt,iChanSol,DoCalcEvP,ThisTime,rms,DoEvP,DoFullPredict,SharedDicoDescriptors = self.work_queue.get()
        #     # print(iAnt,iChanSol,DoCalcEvP,ThisTime,rms,DoEvP,DoFullPredict,SharedDicoDescriptors)
        #     try:
        #         iAnt,iChanSol,DoCalcEvP,ThisTime,rms,DoEvP,DoFullPredict,SharedDicoDescriptors = self.work_queue.get()
        #     except:
        #         break
        #     # #self.e.wait()
        # ####################

            ch0,ch1=self.JonesToVisChanMapping[iChanSol]

            T0=time.time()
            T=ClassTimeIt.ClassTimeIt("  Worker Ant=%2.2i"%iAnt)
            T.disable()
            # if DoCalcEvP:
            #     T.disable()
            # print SharedDicoDescriptors

            if self.SolverType=="CohJones":
                SolverClass=ClassSolverLM.ClassSolverLM
            elif self.SolverType=="KAFCA":
                SolverClass=ClassSolverEKF.ClassSolverEKF

            JM=SolverClass(self.SM,iAnt,PolMode=self.PolMode,
                           PM=self.PM,
                           PM_Compress=self.PM_Compress,
                           SM_Compress=self.SM_Compress,
                           IdSharedMem=self.IdSharedMem,
                           GD=self.GD,
                           ChanSel=(ch0,ch1),
                           SharedDicoDescriptors=SharedDicoDescriptors,
                           **dict(self.ConfigJacobianAntenna))

            T.timeit("ClassJacobianAntenna")

            JM.setDATA_Shared()
            T.timeit("setDATA_Shared")

            G=NpShared.GiveArray("%sSharedGains"%self.IdSharedMem)
            P=NpShared.GiveArray("%sSharedCovariance"%self.IdSharedMem)

            GPrevious=G#NpShared.GiveArray("%sSharedGainsPrevious"%self.IdSharedMem)
            PPrevious=P#NpShared.GiveArray("%sSharedPPrevious"%self.IdSharedMem)

            G0Iter=NpShared.GiveArray("%sSharedGains0Iter"%self.IdSharedMem)

            #Q=NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem)
            evP=NpShared.GiveArray("%sSharedEvolveCovariance"%self.IdSharedMem)
            T.timeit("GiveArray")

            if self.SolverType=="CohJones":
                x,_,InfoNoise=JM.doLMStep(G[iChanSol])
                T.timeit("LM")
                if DoFullPredict: 
                    #print "!!!!!!!!!!!!!!!!!!!"
                    #x[:]=G[iChanSol,iAnt][:]

                    Gc0=G.copy()
                    Gc0[iChanSol,iAnt][:]=x[:]
                    Gc=Gc0.copy()

                    # # Gc0.fill(1.)
                    # NoZeroD=5
                    # Gc.fill(0)
                    # Gc[:,:,NoZeroD,:,:]=Gc0[:,:,NoZeroD,:,:]


                    #Gc.fill(1)
                    JM.PredictOrigFormat(Gc[iChanSol])
                    T.timeit("FullPredict")

                # if not JM.DataAllFlagged:
                #     M=JM.L_JHJ[0]
                #     u,s,v=np.linalg.svd(M)
                #     if np.min(s)<0: stop

                L=[iAnt,
                   iChanSol,
                   x,
                   None,
                   None,
                   InfoNoise,
                   0.,
                   JM.SharedDicoDescriptors["SharedAntennaVis"]]

                
                self.result_queue.put(L)

            elif self.SolverType=="KAFCA":
                #T.disable()
                if DoCalcEvP:
                    evP[iChanSol,iAnt]=JM.CalcMatrixEvolveCov(GPrevious[iChanSol],PPrevious[iChanSol],rms)
                    # if iAnt==51:
                    #     M=(evP[iChanSol,iAnt]!=0)
                    #     x=evP[iChanSol,iAnt][M].ravel()
                    #     print("EVVV",x)
                    #     print("EVVV",x)
                    #     print("EVVV",x)
                    T.timeit("Estimate Evolve")

                # EM=ClassModelEvolution(iAnt,
                #                        StepStart=3,
                #                        WeigthScale=2,
                #                        DoEvolve=False,
                #                        order=1,
                #                        sigQ=0.01)

                EM=ClassModelEvolution(iAnt,iChanSol,
                                       StepStart=0,
                                       WeigthScale=0.5,
                                       DoEvolve=True,
                                       BufferNPoints=10,
                                       sigQ=0.01,IdSharedMem=self.IdSharedMem)
                T.timeit("Init EM")

                Pa=None

                # Ga,Pa=EM.Evolve0(G,P,self.ThisTime)
                # if Ga!=None:
                #     G[iAnt]=Ga
                #     P[iAnt]=Pa

                # ThisP=P[iChanSol].copy()
                # ThisP.fill(0)
                # # print
                # # print ThisP.shape
                # # print
                # na,nd,_=ThisP.shape
                # for iAnt in range(na):
                #     for iDir in range(nd):
                #         ThisP[iAnt,iDir,iDir]=P[iChanSol][iAnt,iDir,iDir]
                # x,Pout,InfoNoise=JM.doEKFStep(G[iChanSol],ThisP,evP[iChanSol],rms,Gains0Iter=G0Iter)
                
                # if iAnt==51:
                #     print("KKKKK",iAnt,G[iChanSol].max(),P[iChanSol].max(),evP[iChanSol].max())
                
                x,Pout,InfoNoise,HasSolved=JM.doEKFStep(G[iChanSol],P[iChanSol],evP[iChanSol],rms,Gains0Iter=G0Iter)
                T.timeit("EKFStep")
                if DoFullPredict: 
                    JM.PredictOrigFormat(G[iChanSol])
                T.timeit("PredictOrigFormat")
                rmsFromData=JM.rmsFromData

                if DoEvP and HasSolved:
                    #Pout0=Pout#.copy()
                    Pa=EM.Evolve0(x,Pout)#,kapa=kapa)
                    
                    # if iAnt==51:
                    #     print("###",iAnt,x.max(),Pa.max(),Pout0.max())
                        
                    T.timeit("Evolve")
                else:
                    Pa=P[iChanSol,iAnt].copy()
                    
                #_,Pa=EM.Evolve(x,Pout,ThisTime)

                if type(Pa)!=type(None):
                    Pout=Pa

                DT=time.time()-T0
                # L=[iAnt,
                #    iChanSol,
                #    x,
                #    Pout,
                #    rmsFromData,
                #    InfoNoise,
                #    DT]
                # self.result_queue.put(L)
                
                self.result_queue.put([iAnt,
                                       iChanSol,
                                       x,
                                       Pout,
                                       rmsFromData,
                                       InfoNoise,
                                       DT,
                                       JM.SharedDicoDescriptors["SharedAntennaVis"]])
