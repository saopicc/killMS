
from ClassJacobianAntenna import ClassJacobianAntenna
import numpy as np
from Array import NpShared

from Data import ClassVisServer
from Sky import ClassSM
from Array import ModLinAlg
import pylab

from Other import MyLogger
log=MyLogger.getLogger("ClassWirtingerSolver")
from Other import ModColor

from Other.progressbar import ProgressBar
            
from Sky.PredictGaussPoints_NumExpr import ClassPredict
from Other import ClassTimeIt
from Other import Counter
from ClassEvolve import ClassModelEvolution


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
                 PolMode="HalfFull",
                 Lambda=1,NIter=20,
                 NCPU=6,
                 SolverType="CohJones",
                 IdSharedMem="",
                 evP_StepStart=0, evP_Step=1,
                 DoPlot=False,
                 DoPBar=True):
        self.IdSharedMem=IdSharedMem
        self.Lambda=Lambda
        self.NCPU=NCPU
        self.DoPBar=DoPBar
        if BeamProps!=None:
            rabeam,decbeam=SM.ClusterCat.ra,SM.ClusterCat.dec
            Mode,TimeMin=BeamProps
            LofarBeam=(Mode,TimeMin,rabeam,decbeam)
            VS.SetBeam(LofarBeam)
        self.DoPlot=DoPlot
        MS=VS.MS
        SM.Calc_LM(MS.rac,MS.decc)
        self.SM=SM
        self.VS=VS
        self.PolMode=PolMode
        self.G=None
        self.NIter=NIter
        #self.SolsList=[]
        self.iCurrentSol=0
        self.SolverType=SolverType
        self.rms=None
        self.rmsFromData=None
        if SolverType=="KAFCA":
           self.NIter=1

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

    def GiveSols(self):
        ind=np.where(self.SolsArray_done==1)[0]
        self.SolsArray_Full.t0[0:ind.size]=self.SolsArray_t0[0:ind.size]
        self.SolsArray_Full.t1[0:ind.size]=self.SolsArray_t1[0:ind.size]
        if self.PolMode=="Scalar":
            self.SolsArray_Full.G[0:ind.size,:,:,0,0]=self.SolsArray_G[0:ind.size,:,:,0,0]
            self.SolsArray_Full.G[0:ind.size,:,:,1,1]=self.SolsArray_G[0:ind.size,:,:,0,0]
        else:                
            self.SolsArray_Full.G[0:ind.size]=self.SolsArray_G[0:ind.size]

        Std=np.array(self.ListStd)
        Max=np.array(self.ListMax)
        Kapa=np.array(self.ListKapa)

        na,nt=Std.shape
        NoiseInfo=np.zeros((na,nt,3))
        NoiseInfo[:,:,0]=Std[:,:]
        NoiseInfo[:,:,1]=np.abs(Max[:,:])
        NoiseInfo[:,:,2]=Kapa[:,:]
        np.save("NoiseInfo",NoiseInfo)

        return self.SolsArray_Full[0:ind.size].copy()

    def InitSol(self,G=None,TestMode=True):
        na=self.VS.MS.na
        nd=self.SM.NDir
        


        if type(G)==type(None):
            if self.PolMode=="Scalar":
                npol=1
                G=np.ones((na,nd,1,1),np.complex128)
            else:
                npol=2
                G=np.zeros((na,nd,2,2),np.complex128)
                G[:,:,0,0]=1
                G[:,:,1,1]=1
            self.HasFirstGuessed=False

        else:
            self.HasFirstGuessed=True
        self.G=G
        #self.G*=0.001
        _,_,npol,_=self.G.shape


        # print "!!!!!!!!!!"
        # self.G+=np.random.randn(*self.G.shape)*0.1#sigP
        
        NSols=np.max([1,1.5*int(self.VS.MS.DTh/(self.VS.TVisSizeMin/60.))])
        
        

        self.SolsArray_t0=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_t1=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_tm=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_done=np.zeros((NSols,),dtype=np.bool8)
        self.SolsArray_G=np.zeros((NSols,na,nd,npol,npol),dtype=np.complex64)

        self.SolsArray_t0=NpShared.ToShared("%sSolsArray_t0"%self.IdSharedMem,self.SolsArray_t0)
        self.SolsArray_t1=NpShared.ToShared("%sSolsArray_t1"%self.IdSharedMem,self.SolsArray_t1)
        self.SolsArray_tm=NpShared.ToShared("%sSolsArray_tm"%self.IdSharedMem,self.SolsArray_tm)
        self.SolsArray_done=NpShared.ToShared("%sSolsArray_done"%self.IdSharedMem,self.SolsArray_done)
        self.SolsArray_G=NpShared.ToShared("%sSolsArray_G"%self.IdSharedMem,self.SolsArray_G)

        self.SolsArray_Full=np.zeros((NSols,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(na,nd,2,2))])
        self.SolsArray_Full=self.SolsArray_Full.view(np.recarray)


        self.G=NpShared.ToShared("%sSharedGains"%self.IdSharedMem,self.G)
        self.InitCovariance()

    def InitCovariance(self,FromG=False,sigP=0.1,sigQ=0.01):
        if self.SolverType!="KAFCA": return
        na=self.VS.MS.na
        nd=self.SM.NDir

        
        _,_,npol,_=self.G.shape
        
        if FromG==False:
            if self.PolMode=="Scalar":
                P=(sigP**2)*np.array([np.diag(np.ones((nd,),np.complex128)) for iAnt in range(na)])
                Q=(sigQ**2)*np.array([np.diag(np.ones((nd,),np.complex128)) for iAnt in range(na)])
            else:
                P=(sigP**2)*np.array([np.diag(np.ones((nd*2*2),np.complex128)) for iAnt in range(na)])
                Q=(sigQ**2)*np.array([np.diag(np.ones((nd*2*2),np.complex128)) for iAnt in range(na)])
        else:

            P=(sigP**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*np.diag(np.ones((nd*npol*npol),np.complex128)) for iAnt in range(na)])
            Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*np.diag(np.ones((nd*npol*npol),np.complex128)) for iAnt in range(na)])
            #P=(sigP**2)*np.array([np.complex128(np.diag(np.abs(self.G[iAnt]).flatten())) for iAnt in range(na)])
            #Q=(sigQ**2)*np.array([np.complex128(np.diag(np.abs(self.G[iAnt]).flatten())) for iAnt in range(na)])


        self.P=P
        self.Q=Q
        self.evP=np.zeros_like(P)
        self.P=NpShared.ToShared("%sSharedCovariance"%self.IdSharedMem,self.P)
        self.Q=NpShared.ToShared("%sSharedCovariance_Q"%self.IdSharedMem,self.Q)
        self.Q_Init=self.Q.copy()
        self.evP=NpShared.ToShared("%sSharedEvolveCovariance"%self.IdSharedMem,self.evP)
        nbuff=10
        self.ListKapa=[[] for iAnt in range(na)]
        self.ListStd=[[] for iAnt in range(na)]
        self.ListMax=[[] for iAnt in range(na)]


    def setNextData(self):
        DATA=self.VS.GiveNextVis()
        if DATA=="EndOfObservation":
            print>>log, ModColor.Str("Reached end of data")
            return "EndOfObservation"
        if DATA=="EndChunk":
            print>>log, ModColor.Str("Reached end of data chunk")
            return "EndChunk"


        ## simul
        #d=self.DATA["data"]
        #self.DATA["data"]+=(self.rms/np.sqrt(2.))*(np.random.randn(*d.shape)+1j*np.random.randn(*d.shape))

        self.rms=-1
        if self.rmsFromData!=None:
            self.rms=self.rmsFromData
        elif self.rmsFromExt!=None:
            self.rms=self.rmsFromExt
        else:
            Dpol=DATA["data"][:,:,1:3]
            Fpol=DATA["flags"][:,:,1:3]
            self.rms=np.std(Dpol[Fpol==0])/np.sqrt(2.)

        print "rms=",self.rms


        return True

    def SetRmsFromExt(self,rms):
        self.rmsFromExt=rms
        

    #################################
    ##          Serial             ## 
    #################################

    def doNextTimeSolve(self):



        if self.G==None:
            self.InitSol()

        ListAntSolve=[i for i in range(self.VS.MS.na) if not(i in self.VS.FlagAntNumber)]
        self.DicoJM={}


        T=ClassTimeIt.ClassTimeIt("WirtingerSolver")
        T.disable()
        while True:
            T.reinit()
            Res=self.setNextData()
            if Res=="EndChunk": break
            T.timeit("read data")

            
            t0,t1=self.VS.CurrentVisTimes_MS_Sec
            self.SolsArray_t0[self.iCurrentSol]=t0
            self.SolsArray_t1[self.iCurrentSol]=t1
            tm=(t0+t1)/2.
            self.SolsArray_tm[self.iCurrentSol]=tm
            ThisTime=tm
            T.timeit("stuff")
            for iAnt in ListAntSolve:
                JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Lambda=self.Lambda,Precision="D",IdSharedMem=self.IdSharedMem)
                T.timeit("JM")
                JM.setDATA_Shared()
                T.timeit("Setdata_Shared")
                self.DicoJM[iAnt]=JM

            T.timeit("Class")

            if (self.CounterEvolveP())&(self.SolverType=="KAFCA")&(self.iCurrentSol>self.EvolvePStepStart):
                print "Evolve0"
                for iAnt in self.DicoJM.keys():
                    JM=self.DicoJM[iAnt]
                    self.evP[iAnt]=JM.CalcMatrixEvolveCov(self.G,self.P,self.rms)

            elif (self.SolverType=="KAFCA")&(self.iCurrentSol<=self.EvolvePStepStart):
                print "Evolve1"
                for iAnt in self.DicoJM.keys():
                    JM=self.DicoJM[iAnt]
                    self.evP[iAnt]=JM.CalcMatrixEvolveCov(self.G,self.P,self.rms)

            T.timeit("Evolve")
            

            for i in range(self.NIter):
                Gnew=self.G.copy()
                if self.SolverType=="KAFCA":
                    Pnew=self.P.copy()
                for iAnt in self.DicoJM.keys():
                    JM=self.DicoJM[iAnt]
                    if self.SolverType=="CohJones":

                        x=JM.doLMStep(self.G)

                    if self.SolverType=="KAFCA":
                        EM=ClassModelEvolution(iAnt,
                                               StepStart=3,
                                               WeigthScale=0.3,
                                               DoEvolve=True,
                                               BufferNPoints=10,
                                               sigQ=0.01,IdSharedMem=self.IdSharedMem)

                        x,P,_=JM.doEKFStep(self.G,self.P,self.evP,self.rms)
                        
                        xe=None

                        Pa=EM.Evolve0(x,P)

                        #xe,Pa=EM.Evolve(x,P,ThisTime)
                        if Pa!=None:
                            P=Pa
                        if xe!=None:
                            x=xe


                        Pnew[iAnt]=P

                    Gnew[iAnt]=x
                    T.timeit("SolveAnt %i"%iAnt)
                

                pylab.figure(1)
                pylab.clf()
                pylab.plot(np.abs(Gnew.flatten()))
                if self.SolverType=="KAFCA":
                    sig=np.sqrt(np.array([np.diag(Pnew[iAnt]) for iAnt in range(self.VS.MS.na)]).flatten())
                    pylab.plot(np.abs(Gnew.flatten())+sig,color="black",ls="--")
                    pylab.plot(np.abs(Gnew.flatten())-sig,color="black",ls="--")
                    self.P[:]=Pnew[:]
                pylab.plot(np.abs(self.G.flatten()))
                pylab.ylim(0,2)
                pylab.draw()
                pylab.show(False)
                self.G[:]=Gnew[:]

                T.timeit("Plot")


            self.SolsArray_done[self.iCurrentSol]=1
            self.SolsArray_G[self.iCurrentSol][:]=self.G[:]

            
            self.iCurrentSol+=1
        return True



    # #################################
    # ###        Parallel           ###
    # #################################
    
    
    
    def doNextTimeSolve_Parallel(self,OnlyOne=False,SkipMode=False):

        




        ListAntSolve=[i for i in range(self.VS.MS.na) if not(i in self.VS.FlagAntNumber)]

        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()




        workerlist=[]
        NCPU=self.NCPU

        import time
        
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
                    


        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        for ii in range(NCPU):
             
            W=WorkerAntennaLM(work_queue, result_queue,self.SM,self.PolMode,self.Lambda,self.SolverType,self.IdSharedMem)#,args=(e,))
            workerlist.append(W)
            workerlist[ii].start()



        ##############################

        T0,T1=self.VS.TimeMemChunkRange_sec[0],self.VS.TimeMemChunkRange_sec[1]
        DT=(T1-T0)
        dt=self.VS.TVisSizeMin*60.
        nt=int(DT/float(dt))+1
        

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Solving ", HeaderSize=10,TitleSize=13)
        if not(self.DoPBar): pBAR.disable()
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,nt))
        NDone=0
        iiCount=0
        while True:
            Res=self.setNextData()
            if Res=="EndChunk": break
            if SkipMode:
                print iiCount
                iiCount+=1
                if iiCount<170: continue

            t0,t1=self.VS.CurrentVisTimes_MS_Sec
            self.SolsArray_t0[self.iCurrentSol]=t0
            self.SolsArray_t1[self.iCurrentSol]=t1
            tm=(t0+t1)/2.
            self.SolsArray_tm[self.iCurrentSol]=tm
            


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


            Gold=self.G.copy()
            for LMIter in range(NIter):
                #print
                # for EKF
                DoCalcEvP=False
                if (self.CounterEvolveP())&(self.SolverType=="KAFCA")&(self.iCurrentSol>self.EvolvePStepStart):
                    DoCalcEvP=True
                elif (self.SolverType=="KAFCA")&(self.iCurrentSol<=self.EvolvePStepStart):
                    DoCalcEvP=True

                #########

                for iAnt in ListAntSolve:
                    work_queue.put((iAnt,DoCalcEvP,tm,self.rms))
 
                rmsFromDataList=[]
                while iResult < NJobs:
                    iAnt,G,P,rmsFromData,InfoNoise = result_queue.get()
                    if rmsFromData!=None:
                        rmsFromDataList.append(rmsFromData)
                    
                    self.G[iAnt][:]=G[:]
                    if type(P)!=type(None):
                        self.P[iAnt,:]=P[:]

                    kapa=InfoNoise["kapa"]
                    self.ListStd[iAnt].append(InfoNoise["std"])
                    self.ListMax[iAnt].append(InfoNoise["max"])


                    iResult+=1
                    if kapa!=None:
                        if kapa==-1.:
                            if len(self.ListKapa[iAnt])>0:
                                kapa=self.ListKapa[iAnt][-1]
                            else:
                                kapa=1.
                        self.ListKapa[iAnt].append(kapa)
                        dt=1.
                        TraceResidList=self.ListKapa[iAnt]
                        x=np.arange(len(TraceResidList))
                        expW=np.exp(-x/dt)[::-1]
                        expW/=np.sum(expW)
                        kapaW=np.sum(expW*np.array(TraceResidList))
                        #self.Q[iAnt]=(kapaW**2)*self.Q_Init[iAnt]
                        self.Q[iAnt]=(kapaW)*self.Q_Init[iAnt]
                        #print iAnt,kapa,kapaW
                        #sig=np.sqrt(np.abs(np.array([np.diag(self.P[i]) for i in [iAnt]]))).flatten()
                        #print sig
                        

                if len(rmsFromDataList)>0:
                    self.rmsFromData=np.min(rmsFromDataList)
                iResult=0


                if self.DoPlot:
                    AntPlot=np.array(ListAntSolve)
                    pylab.clf()
                    pylab.plot(np.abs(self.G[AntPlot].flatten()))
                    pylab.plot(np.abs(Gold[AntPlot].flatten()))
                    if self.SolverType=="KAFCA":
                        sig=np.sqrt(np.abs(np.array([np.diag(self.P[i]) for i in ListAntSolve]))).flatten()
                        pylab.plot(np.abs(self.G[AntPlot].flatten())+sig,color="black",ls="--")
                        pylab.plot(np.abs(self.G[AntPlot].flatten())-sig,color="black",ls="--")
                    #pylab.ylim(0,2)
                    pylab.draw()
                    pylab.show(False)
                    pylab.pause(0.1)


            NDone+=1
            intPercent=int(100*  NDone / float(nt))

            pBAR.render(intPercent, '%4i/%i' % (NDone,nt))
                
            
            self.SolsArray_done[self.iCurrentSol]=1
            self.SolsArray_G[self.iCurrentSol][:]=self.G[:]
            self.iCurrentSol+=1
            
            if OnlyOne: break


 
        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

            
        return True



 






#======================================
import multiprocessing
class WorkerAntennaLM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,SM,PolMode,Lambda,SolverType,IdSharedMem,**kwargs):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM
        self.PolMode=PolMode
        self.Lambda=Lambda
        self.SolverType=SolverType
        self.IdSharedMem=IdSharedMem
        #self.DoCalcEvP=DoCalcEvP
        #self.ThisTime=ThisTime
        #self.e,=kwargs["args"]

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            try:
                iAnt,DoCalcEvP,ThisTime,rms = self.work_queue.get()
            except:
                break
            #self.e.wait()

            JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Lambda=self.Lambda,IdSharedMem=self.IdSharedMem)
            JM.setDATA_Shared()

            G=NpShared.GiveArray("%sSharedGains"%self.IdSharedMem)
            P=NpShared.GiveArray("%sSharedCovariance"%self.IdSharedMem)
            #Q=NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem)
            evP=NpShared.GiveArray("%sSharedEvolveCovariance"%self.IdSharedMem)

            if self.SolverType=="CohJones":
                x=JM.doLMStep(G)
                self.result_queue.put([iAnt,x,None,None,None])
            elif self.SolverType=="KAFCA":
                T=ClassTimeIt.ClassTimeIt()
                T.disable()
                if DoCalcEvP:
                    evP[iAnt]=JM.CalcMatrixEvolveCov(G,P,rms)
                    T.timeit("Estimate Evolve")

                # EM=ClassModelEvolution(iAnt,
                #                        StepStart=3,
                #                        WeigthScale=2,
                #                        DoEvolve=False,
                #                        order=1,
                #                        sigQ=0.01)

                EM=ClassModelEvolution(iAnt,
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

                x,Pout,InfoNoise=JM.doEKFStep(G,P,evP,rms)
                T.timeit("EKFStep")
                rmsFromData=JM.rmsFromData
                Pa=EM.Evolve0(x,Pout)#,kapa=kapa)
                T.timeit("Evolve")
                #_,Pa=EM.Evolve(x,Pout,ThisTime)

                if type(Pa)!=type(None):
                    Pout=Pa

                self.result_queue.put([iAnt,x,Pout,rmsFromData,InfoNoise])
