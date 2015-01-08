
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
import NpShared
from progressbar import ProgressBar

from PredictGaussPoints_NumExpr import ClassPredict

def test():


    ReadColName="DATA"
    WriteColName="CORRECTED_DATA"
    SM=ClassSM.ClassSM("../TEST/ModelRandom00.txt.npy",
                       killdirs=["c0s0."],invert=False)
    
    VS=ClassVisServer.ClassVisServer("../TEST/0000.MS",ColName=ReadColName,TVisSizeMin=10)
    
    #LM=ClassLM(VS,SM,PolMode="Scalar")
    LM=ClassLM(VS,SM,PolMode="HalfFull")
    # LM.doNextTimeSolve()
    #LM.doNextTimeSolve_Parallel()
    #return
    PM=ClassPredict()
    SM=LM.SM

    while True:
        Res=LM.doNextTimeSolve_Parallel()
        if Res==True:
            continue
        else:
            # substract
            
            Jones={}
            Jones["t0"]=LM.Sols.t0
            Jones["t1"]=LM.Sols.t1
            nt,na,nd,_,_=LM.Sols.G.shape
            G=np.swapaxes(LM.Sols.G,1,2).reshape((nt,nd,na,1,2,2))
            Jones["Beam"]=G
            Jones["BeamH"]=ModLinAlg.BatchH(G)

            SM.SelectSubCat(SM.SourceCat.kill==1)
            PredictData=PM.predictKernelPolCluster(LM.VS.ThisDataChunk,LM.SM,ApplyTimeJones=Jones)
            SM.RestoreCat()

            LM.VS.ThisDataChunk["data"]-=PredictData
            LM.VS.MS.data=LM.VS.ThisDataChunk["data"]
            LM.VS.MS.SaveVis(Col=WriteColName)

        if Res=="EndOfObservation":
            break

class ClassLM():

    def __init__(self,VS,SM,BeamProps=None,PolMode="HalfFull",Lambda=1,NIter=20):
        self.Lambda=Lambda
        if BeamProps!=None:
            rabeam,decbeam=SM.ClusterCat.ra,SM.ClusterCat.dec
            Mode,TimeMin=BeamProps
            LofarBeam=(Mode,TimeMin,rabeam,decbeam)
            VS.SetBeam(LofarBeam)

        MS=VS.MS
        SM.Calc_LM(MS.rac,MS.decc)
        self.SM=SM
        self.VS=VS
        self.PolMode=PolMode
        self.G=None
        self.NIter=NIter

        #### Solutions
        self.NSols=self.VS.TimesVisMin.size-1
        na=self.VS.MS.na
        nd=self.SM.NDir
        self.Sols=np.zeros((self.NSols,),dtype=[("t0",float),("t1",float),("G",np.complex64,(na,nd,2,2))])
        self.Sols=self.Sols.view(np.recarray)
        self.iCurrentSol=0
        
    def InitSol(self):
        na=self.VS.MS.na
        nd=self.SM.NDir
        if self.PolMode=="Scalar":
            G=np.ones((na,nd,1,1),np.complex64)
        else:
            G=np.zeros((na,nd,2,2),np.complex64)
            G[:,:,0,0]=1
            G[:,:,1,1]=1
        self.G=G
        self.G+=np.random.randn(*self.G.shape)*1


    # #################################
    # ###        Parallel           ###
    # #################################
    
    
    def doNextTimeSolve_Parallel(self):
        DATA=self.VS.GiveNextVis()
        if DATA=="EndOfObservation":
            print>>log, ModColor.Str("Reached end of data")
            return "EndOfObservation"
        if DATA=="EndChunk":
            print>>log, ModColor.Str("Reached end of data chunk")
            return "EndChunk"
        self.DATA=DATA

        t0,t1=self.VS.CurrentVisTimes
        self.Sols.t0[self.iCurrentSol]=t0
        self.Sols.t1[self.iCurrentSol]=t1


        if self.G==None:
            self.InitSol()
            self.G=NpShared.ToShared("SharedGains",self.G)

        ListAntSolve=range(self.VS.MS.na)

        work_queue = multiprocessing.Queue()
        EventList=[multiprocessing.Event() for i in range(self.NIter)]
        e=EventList[0]


        NJobs=len(ListAntSolve)
        for iAnt in ListAntSolve:
             work_queue.put((iAnt))

        result_queue = multiprocessing.Queue()

        workerlist=[]
        NCPU=6
        import time
        pylab.figure(1)
        pylab.clf()
        pylab.plot(np.abs(self.G.flatten()))
        pylab.ylim(-2,2)
        pylab.draw()
        pylab.show(False)

        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt()
        for ii in range(NCPU):
            
            W=WorkerAntennaLM(work_queue, result_queue,self.SM,self.PolMode,self.Lambda)#,args=(e,))
            workerlist.append(W)
            workerlist[ii].start()
            # time.sleep(2)

        #print ModColor.Str(" Pealing in [%-.2f->%-.2f h]"%(T0,T1),Bold=False)
        toolbar_width = 50
        
        #pBAR= ProgressBar('white', block='=', empty=' ',Title="Solving")
        #pBAR.render(0, '%i/%i' % (0,NJobs-1.))

        #e=EventList[0]
        #time.sleep(3)
        #e.set()

        results = []
        lold=len(results)
        iResult=0
        for LMIter in range(self.NIter):
            while iResult < NJobs:
                iAnt,G = result_queue.get()
                self.G[iAnt][:]=G[:]
                iResult+=1
            iResult=0
            # pylab.plot(np.abs(self.G.flatten()))
            # pylab.ylim(-2,2)
            # pylab.draw()
            for iAnt in ListAntSolve:
                work_queue.put((iAnt))
            
            # if len(results)>lold:
            #     lold=len(results)
            #     #pBAR.render(int(100* float(lold) / (NJobs-1.)), '%i/%i' % (lold,NJobs-1.))
                
            # # if ii/10.==1:
            # #     pylab.plot(np.abs(self.G.flatten()))
            # #     pylab.draw()
            # #     ii=0
            # # ii+=1


 
        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        if self.PolMode=="Scalar":
            self.Sols.G[self.iCurrentSol][:,:,0,0]=self.G[:,:,0,0]
            self.Sols.G[self.iCurrentSol][:,:,1,1]=self.G[:,:,0,0]
        else:
            self.Sols.G[self.iCurrentSol][:]=self.G[:]
            
        self.iCurrentSol+=1
        return True



 
    # results = []
    # lold=len(results)
    # while len(results) < len(jobs):
    #     result = result_queue.get()
    #     results.append(result)
    #     if len(results)>lold:
    #         lold=len(results)
    #         pBAR.render(int(100* float(lold) / (len(ss)-1.)), '%i/%i' % (lold,len(ss)-1.))

    # for ii in range(NCPU):
    #     workerlist[ii].shutdown()
    #     workerlist[ii].terminate()
    #     workerlist[ii].join()


    #################################
    ##          Serial             ## 
    #################################

    def doNextTimeSolve(self):
        DATA=self.VS.GiveNextVis()
        if DATA==None:
            print>>log, ModColor.Str("Reached end of data")
            return False
        self.DATA=DATA

        t0,t1=self.VS.CurrentVisTimes
        self.Sols.t0[self.iCurrentSol]=t0
        self.Sols.t1[self.iCurrentSol]=t1


        if self.G==None:
            self.InitSol()

        ListAntSolve=range(self.VS.MS.na)
        self.DicoJM={}
        for iAnt in ListAntSolve:
            JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Lambda=self.Lambda)
            JM.setDATA(DATA)
            self.DicoJM[iAnt]=JM

        for i in range(self.NIter):
            for iAnt in self.DicoJM.keys():
                JM=self.DicoJM[iAnt]
                x=JM.doLMStep(self.G)
                self.G[iAnt]=x

            pylab.figure(1)
            pylab.clf()
            pylab.plot(np.abs(self.G.flatten()))
            pylab.ylim(-2,2)
            pylab.draw()
            pylab.show(False)


        if self.PolMode=="Scalar":
            self.Sols.G[self.iCurrentSol][:,:,0,0]=self.G[:,:,0,0]
            self.Sols.G[self.iCurrentSol][:,:,1,1]=self.G[:,:,0,0]
        else:
            self.Sols.G[self.iCurrentSol][:]=self.G[:]
            
        self.iCurrentSol+=1
        return True




#======================================
import multiprocessing
class WorkerAntennaLM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,SM,PolMode,Lambda,**kwargs):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM
        self.PolMode=PolMode
        self.Lambda=Lambda
        #self.e,=kwargs["args"]

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            try:
                iAnt = self.work_queue.get()
            except:
                break
            #self.e.wait()

            JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Lambda=self.Lambda)
            JM.setDATA_Shared()

            G=NpShared.GiveArray("SharedGains")
            x=JM.doLMStep(G)
            self.result_queue.put([iAnt,x])
