
from ClassJacobianAntenna import ClassJacobianAntenna
import numpy as np
from Array import NpShared

from Data import ClassVisServer
from Sky import ClassSM
from Array import ModLinAlg
import pylab

from Other import MyLogger
log=MyLogger.getLogger("ClassLM")
from Other import ModColor

from Other.progressbar import ProgressBar
            
from Sky.PredictGaussPoints_NumExpr import ClassPredict
from Other import ClassTimeIt

def test():


    ReadColName="DATA"
    WriteColName="CORRECTED_DATA"
    SM=ClassSM.ClassSM("/media/tasse/data/HyperCal2/test/ModelRandom00.txt.npy",
                       killdirs=["c0s0."],invert=False)
    
    VS=ClassVisServer.ClassVisServer("/media/6B5E-87D0/MS/SimulTec/Pointing00/MS/0000.MS",ColName=ReadColName,TVisSizeMin=2,TChunkSize=.1,AddNoiseJy=10.)
    
    #LM=ClassWirtingerSolver(VS,SM,PolMode="Scalar",NIter=1,SolverType="EKF")#20)
    LM=ClassWirtingerSolver(VS,SM,PolMode="Scalar",NIter=10,SolverType="KAFCA")#"CohJones")#"KAFCA")
    # LM.doNextTimeSolve()
    #LM.doNextTimeSolve_Parallel()
    #return
    PM=ClassPredict()
    SM=LM.SM
    LM.InitSol()


    while True:
        Res=LM.setNextData()
        if Res==True:
            #print Res,VS.CurrentVisTimes_SinceStart_Minutes
            # LM.doNextTimeSolve_Parallel()
            LM.doNextTimeSolve()
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

    def __init__(self,VS,SM,BeamProps=None,PolMode="HalfFull",
                 Lambda=1,NIter=20,NCPU=6,SolverType="CohJones"):
        self.Lambda=Lambda
        self.NCPU=NCPU
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
        self.SolsList=[]
        self.iCurrentSol=0
        self.SolverType=SolverType
        if SolverType=="KAFCA":
           self.NIter=1

    def AppendEmptySol(self):
        #### Solutions
        # self.NSols=self.VS.TimesVisMin.size-1
        na=self.VS.MS.na
        nd=self.SM.NDir
        Sol=np.zeros((1,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(na,nd,2,2))])
        self.SolsList.append(Sol.view(np.recarray))

    def GiveSols(self):
        self.SolsArray=np.concatenate(self.SolsList)
        self.SolsArray=self.SolsArray.view(np.recarray)
        return self.SolsArray

    def InitSol(self,TestMode=True):
        na=self.VS.MS.na
        nd=self.SM.NDir
        sigP=0.1
        if self.PolMode=="Scalar":
            G=np.ones((na,nd,1,1),np.complex128)
            P=(sigP**2)*np.array([np.diag(np.ones((nd,),np.complex128)) for iAnt in range(na)])
        else:
            G=np.zeros((na,nd,2,2),np.complex128)
            G[:,:,0,0]=1
            G[:,:,1,1]=1
            P=(sigP**2)*np.array([np.diag(np.ones((nd*2*2),np.complex128)) for iAnt in range(na)])
            
        self.G=G
        self.G+=np.random.randn(*self.G.shape)*sigP
        if TestMode:
            self.G+=np.random.randn(*self.G.shape)*sigP
        #self.G[5]+=np.random.randn(*self.G[5].shape)*sigP

        #self.G[5,0,1,0]=1

        #self.G.fill(1)
        self.P=P
        Npars=self.G[0].size


        self.G=NpShared.ToShared("SharedGains",self.G)
        self.P=NpShared.ToShared("SharedCovariance",self.P)

    def setNextData(self):
        DATA=self.VS.GiveNextVis()
        if DATA=="EndOfObservation":
            print>>log, ModColor.Str("Reached end of data")
            return "EndOfObservation"
        if DATA=="EndChunk":
            print>>log, ModColor.Str("Reached end of data chunk")
            return "EndChunk"

        self.DATA=DATA

        ## simul
        #d=self.DATA["data"]
        #self.DATA["data"]+=(self.rms/np.sqrt(2.))*(np.random.randn(*d.shape)+1j*np.random.randn(*d.shape))

        self.DATA["data"].shape
        Dpol=self.DATA["data"][:,:,1:3]
        Fpol=self.DATA["flags"][:,:,1:3]
        self.rms=np.std(Dpol[Fpol==0])
        self.rms=np.max([self.rms,1e-2])
        #self.rms=np.min(self.rmsPol)
        
        rms=self.rms*1000
        #print>>log, "Estimated rms = %7.2f mJy"%(rms)

        #np.savez("EKF.npz",data=self.DATA["data"],G=self.G)
        #stop
        
        # D=np.load("EKF.npz")
        # self.DATA["data"]=D["data"]
        # self.G=D["G"]

        return True

    #################################
    ##          Serial             ## 
    #################################

    def doNextTimeSolve(self):
        # DATA=self.VS.GiveNextVis()
        # if DATA==None:
        #     print>>log, ModColor.Str("Reached end of data")
        #     return False
        # self.DATA=DATA

        self.AppendEmptySol()
        t0,t1=self.VS.CurrentVisTimes_MS_Sec
        self.SolsList[self.iCurrentSol].t0=t0
        self.SolsList[self.iCurrentSol].t1=t1


        if self.G==None:
            self.InitSol()

        ListAntSolve=range(self.VS.MS.na)
        self.DicoJM={}
        for iAnt in ListAntSolve:
            JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Lambda=self.Lambda,Precision="D")
            #JM.setDATA(DATA)
            JM.setDATA_Shared()
            self.DicoJM[iAnt]=JM


        for i in range(self.NIter):
            Gnew=self.G.copy()
            Pnew=self.P.copy()
            for iAnt in self.DicoJM.keys():
                JM=self.DicoJM[iAnt]
                if self.SolverType=="CohJones":
                    x=JM.doLMStep(self.G)

                if self.SolverType=="KAFCA":
                    x,P=JM.doEKFStep(self.G,self.P,self.rms)
                    Pnew[iAnt]=P

                Gnew[iAnt]=x

            sig=np.sqrt(np.array([np.diag(Pnew[i]) for iAnt in range(self.VS.MS.na)]).flatten())
            pylab.figure(1)
            pylab.clf()
            pylab.plot(np.abs(Gnew.flatten()))
            pylab.plot(np.abs(Gnew.flatten())+sig,color="black",ls="--")
            pylab.plot(np.abs(Gnew.flatten())-sig,color="black",ls="--")
            pylab.plot(np.abs(self.G.flatten()))
            pylab.ylim(0,2)
            pylab.draw()
            pylab.show(False)
            self.G[:]=Gnew[:]
            self.P[:]=Pnew[:]



        # if self.PolMode=="Scalar":
        #     self.Sols.G[self.iCurrentSol][0][:,:,0,0]=self.G[:,:,0,0]
        #     self.Sols.G[self.iCurrentSol][:,:,1,1]=self.G[:,:,0,0]
        # else:
        #     self.Sols.G[self.iCurrentSol][:]=self.G[:]

        if self.PolMode=="Scalar":
            self.SolsList[self.iCurrentSol].G[0][:,:,0,0]=self.G[:,:,0,0]
            self.SolsList[self.iCurrentSol].G[0][:,:,1,1]=self.G[:,:,0,0]
        else:
            self.SolsList[self.iCurrentSol].G[0][:]=self.G[:]
            
        self.iCurrentSol+=1
        return True



    # #################################
    # ###        Parallel           ###
    # #################################
    
    
    
    def doNextTimeSolve_Parallel(self):
        self.AppendEmptySol()
        t0,t1=self.VS.CurrentVisTimes_MS_Sec
        self.SolsList[self.iCurrentSol].t0=t0
        self.SolsList[self.iCurrentSol].t1=t1




        ListAntSolve=range(self.VS.MS.na)

        work_queue = multiprocessing.Queue()
        EventList=[multiprocessing.Event() for i in range(self.NIter)]
        e=EventList[0]


        NJobs=len(ListAntSolve)
        for iAnt in ListAntSolve:
             work_queue.put((iAnt))

        result_queue = multiprocessing.Queue()

        workerlist=[]
        NCPU=self.NCPU

        import time
        
        # pylab.figure(1)
        # pylab.clf()
        # pylab.plot(np.abs(self.G.flatten()))
        # pylab.ylim(-2,2)
        # pylab.draw()
        # pylab.show(False)

        T=ClassTimeIt.ClassTimeIt()
        for ii in range(NCPU):
            
            W=WorkerAntennaLM(work_queue, result_queue,self.SM,self.PolMode,self.Lambda,self.SolverType,self.rms)#,args=(e,))
            workerlist.append(W)
            workerlist[ii].start()
            # time.sleep(2)

        #print ModColor.Str(" Pealing in [%-.2f->%-.2f h]"%(T0,T1),Bold=False)

        NTotJobs=NJobs*self.NIter

        t0_min,t1_min=self.VS.CurrentVisTimes_SinceStart_Minutes
        pBAR= ProgressBar('white', width=30, block='=', empty=' ',Title="Solving in [%.1f, %.1f] min"%(t0_min,t1_min), TitleSize=50)
        pBAR.render(0, '%i/%i' % (0,NTotJobs-1.))

        #e=EventList[0]
        #time.sleep(3)
        #e.set()


        lold=0
        iResult=0
        NDone=0
        for LMIter in range(self.NIter):
            while iResult < NJobs:
                iAnt,G,P = result_queue.get()
                self.G[iAnt][:]=G[:]
                if P!=None:
                    self.P[iAnt,:]=P[:]
                iResult+=1
                NDone+=1
                pBAR.render(int(100* float(NDone-1) / (NTotJobs-1.)), '%4i/%i' % (NDone-1,NTotJobs-1.))
            iResult=0
            
            # pylab.clf()
            # pylab.plot(np.abs(self.G.flatten()))
            # pylab.ylim(-2,2)
            # pylab.draw()
            # pylab.show(False)

            for iAnt in ListAntSolve:
                work_queue.put((iAnt))
            
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
            self.SolsList[self.iCurrentSol].G[0][:,:,0,0]=self.G[:,:,0,0]
            self.SolsList[self.iCurrentSol].G[0][:,:,1,1]=self.G[:,:,0,0]
        else:
            self.SolsList[self.iCurrentSol].G[0][:]=self.G[:]
            
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






#======================================
import multiprocessing
class WorkerAntennaLM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,SM,PolMode,Lambda,SolverType,rms,**kwargs):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM
        self.PolMode=PolMode
        self.Lambda=Lambda
        self.SolverType=SolverType
        self.rms=rms
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
            P=NpShared.GiveArray("SharedCovariance")
            if self.SolverType=="CohJones":
                x=JM.doLMStep(G)
                self.result_queue.put([iAnt,x,None])
            elif self.SolverType=="KAFCA":
                x,Pout=JM.doEKFStep(G,P,self.rms)
                self.result_queue.put([iAnt,x,Pout])
