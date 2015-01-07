
from ClassJacobianAntenna import ClassJacobianAntenna
import numpy as np
import NpShared
from PredictGaussPoints_NumExpr import ClassPredict

import ClassVisServer
import ClassSM
import ModLinAlg
import pylab

import MyLogger
log=MyLogger.getLogger("ClassLM")
import ModColor

def test():
    # LM=ClassLM("../TEST/0000.MS","../TEST/ModelRandom00.txt.npy",PolMode="Scalar")
    LM=ClassLM("../TEST/0000.MS","../TEST/ModelRandom00.txt.npy",PolMode="HalfFull")
    LM.doNextTimeSolve()

class ClassLM():

    def __init__(self,MSName,SMName,BeamProps=None,PolMode="HalfFull",Lambda=1,NIter=20):
        VS=ClassVisServer.ClassVisServer(MSName,TVisSizeMin=20)
        SM=ClassSM.ClassSM(SMName)

        if BeamProps!=None:
            rabeam,decbeam=SM.ClusterCat.ra,SM.ClusterCat.dec
            Mode,TimeMin=BeamProps
            LofarBeam=(Mode,TimeMin,rabeam,decbeam)
            VS.SetBeam(LofarBeam)

        MS=VS.MS
        SM.Calc_LM(MS.rac,MS.decc)
        self.SM=SM
        self.VS=VS
        self.DicoJM={}
        self.PolMode=PolMode
        for iAnt in range(MS.na):
            self.DicoJM[iAnt]=ClassJacobianAntenna(SM,iAnt,PolMode=PolMode,Lambda=Lambda)
        self.G=None
        self.NIter=NIter

        #### Solutions
        self.NSols=self.VS.TimesVisMin.size-1
        na=self.VS.MS.na
        nd=self.SM.NDir
        if self.PolMode=="Scalar":
            npol=1
        else:
            npol=2
        self.Sols=np.zeros((self.NSols,),dtype=[("t0",float),("t1",float),("G",np.complex64,(na,nd,npol,npol))])
        self.Sols=self.Sols.view(np.recarray)
        self.iCurrentSol=0

    def doNextTimeSolve(self):
        DATA=self.VS.GiveNextVis()
        if DATA==None:
            print>>log, ModColor.Str("Reached end of data")
            return
        self.DATA=DATA

        t0,t1=self.VS.CurrentVisTimes
        self.Sols.t0[self.iCurrentSol]=t0
        self.Sols.t1[self.iCurrentSol]=t1

        for iAnt in self.DicoJM.keys():
            JM=self.DicoJM[iAnt]
            JM.setDATA(DATA)
            JM.CalcKernelMatrix()
        if self.G==None:
            self.InitSol()
        for i in range(self.NIter):
            self.doLMStep()
        self.Sols.G[self.iCurrentSol]=self.G.copy()
        self.iCurrentSol+=1

    def InitSol(self):
        na=self.VS.MS.na
        nd=self.SM.NDir
        if self.PolMode=="Scalar":
            G=np.ones((na,nd,1,1),np.complex64)
        else:
            G=np.zeros((na,nd,2,2),np.complex64)
            G[:,:,0,0]=1
            G[:,:,1,1]=1
            Sols=np.zeros((self.NSols,na,nd,2,2),np.complex64)
        self.G=G
        self.G+=np.random.randn(*self.G.shape)*1

    def doLMStep(self):

        for iAnt in self.DicoJM.keys():
            JM=self.DicoJM[iAnt]
            x=JM.doLMStep(self.G)
            self.G[iAnt]=x
        
        # pylab.figure(1)
        # pylab.clf()
        # pylab.plot(np.abs(self.G.flatten()))
        # pylab.ylim(-2,2)
        # pylab.draw()
        # pylab.show(False)

        # pylab.figure(3)
        # pylab.clf()
        # pylab.subplot(1,3,1)
        # pylab.imshow(np.abs(JM.Jacob)[0:20],interpolation="nearest")
        # pylab.subplot(1,3,2)
        # pylab.imshow(np.abs(JM.JHJ),interpolation="nearest")
        # pylab.subplot(1,3,3)
        # pylab.imshow(np.abs(JM.JHJinv),interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
    

