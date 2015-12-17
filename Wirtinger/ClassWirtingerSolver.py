
from ClassJacobianAntenna import ClassJacobianAntenna
import numpy as np
from killMS2.Array import NpShared

from killMS2.Data import ClassVisServer
#from Sky import ClassSM
from killMS2.Array import ModLinAlg
import matplotlib.pyplot as pylab

from killMS2.Other import MyLogger
log=MyLogger.getLogger("ClassWirtingerSolver")
from killMS2.Other import ModColor

from killMS2.Other.progressbar import ProgressBar
            
#from Sky.PredictGaussPoints_NumExpr import ClassPredict
from killMS2.Other import ClassTimeIt
from killMS2.Other import Counter
from ClassEvolve import ClassModelEvolution
import time

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
                 ConfigJacobianAntenna={},TypeRMS="GlobalData"):
        self.DType=np.complex128
        self.TypeRMS=TypeRMS
        self.IdSharedMem=IdSharedMem
        self.ConfigJacobianAntenna=ConfigJacobianAntenna
        self.Lambda=Lambda
        self.NCPU=NCPU
        self.DoPBar=DoPBar
        self.GD=GD
        self.Q=None
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
        self.G=None
        self.NIter=NIter
        #self.SolsList=[]
        self.iCurrentSol=0
        self.SolverType=SolverType
        self.rms=None
        self.rmsFromData=None

        # if SolverType=="KAFCA":
        #     print>>log, ModColor.Str("niter=%i"%self.NIter)
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
            self.SolsArray_Full.G[0:ind.size,:,:,0,0]=self.SolsArray_G[0:ind.size,:,:,0,0]
            self.SolsArray_Full.G[0:ind.size,:,:,1,1]=self.SolsArray_G[0:ind.size,:,:,0,0]
        elif self.PolMode=="IDiag":
            self.SolsArray_Full.G[0:ind.size,:,:,0,0]=self.SolsArray_G[0:ind.size,:,:,0,0]
            self.SolsArray_Full.G[0:ind.size,:,:,1,1]=self.SolsArray_G[0:ind.size,:,:,1,0]
        else:                
            self.SolsArray_Full.G[0:ind.size]=self.SolsArray_G[0:ind.size]



        if SaveStats:
            ListStd=[l for l in self.ListStd if len(l)>0]
            Std=np.array(ListStd)
            ListMax=[l for l in self.ListMax if len(l)>0]
            Max=np.array(ListMax)
            
            ListKapa=[l for l in self.ListKeepKapa if len(l)>0]
            Kapa=np.array(ListKapa)
            na,nt=Std.shape
            NoiseInfo=np.zeros((na,nt,3))
            NoiseInfo[:,:,0]=Std[:,:]
            NoiseInfo[:,:,1]=np.abs(Max[:,:])
            NoiseInfo[:,:,2]=Kapa[:,:]
            
            StatFile="NoiseInfo.npy"
            print>>log, "Saving statistics in %s"%StatFile
            np.save(StatFile,NoiseInfo)


        Sols=self.SolsArray_Full[0:ind.size].copy()
        Sols.t1[-1]+=1e3
        Sols.t0[0]-=1e3

        return Sols

    def InitSol(self,G=None,TestMode=True):
        na=self.VS.MS.na
        nd=self.SM.NDir
        

        if type(G)==type(None):
            if self.PolMode=="Scalar":
                G=np.ones((na,nd,1,1),self.DType)
            elif self.PolMode=="IDiag":
                G=np.ones((na,nd,2,1),self.DType)
            else:
                G=np.zeros((na,nd,2,2),self.DType)
                G[:,:,0,0]=1
                G[:,:,1,1]=1
            self.HasFirstGuessed=False

        else:
            self.HasFirstGuessed=True
        self.G=G
        #self.G*=0.001
        _,_,npolx,npoly=self.G.shape


        #print "!!!!!!!!!!"
        #self.G+=np.random.randn(*self.G.shape)*1#sigP
        
        NSols=np.max([1,1.5*int(self.VS.MS.DTh/(self.VS.TVisSizeMin/60.))])
        
        

        self.SolsArray_t0=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_t1=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_tm=np.zeros((NSols,),dtype=np.float64)
        self.SolsArray_done=np.zeros((NSols,),dtype=np.bool8)
        self.SolsArray_G=np.zeros((NSols,na,nd,npolx,npoly),dtype=np.complex64)
        self.SolsArray_Stats=np.zeros((NSols,na,4),dtype=np.float32)

        self.SolsArray_t0=NpShared.ToShared("%sSolsArray_t0"%self.IdSharedMem,self.SolsArray_t0)
        self.SolsArray_t1=NpShared.ToShared("%sSolsArray_t1"%self.IdSharedMem,self.SolsArray_t1)
        self.SolsArray_tm=NpShared.ToShared("%sSolsArray_tm"%self.IdSharedMem,self.SolsArray_tm)
        self.SolsArray_done=NpShared.ToShared("%sSolsArray_done"%self.IdSharedMem,self.SolsArray_done)
        self.SolsArray_G=NpShared.ToShared("%sSolsArray_G"%self.IdSharedMem,self.SolsArray_G)
        self.SolsArray_Full=np.zeros((NSols,),dtype=[("t0",np.float64),
                                                     ("t1",np.float64),
                                                     ("G",np.complex64,(na,nd,2,2)),
                                                     ("Stats",np.float32,(na,4))])
        self.SolsArray_Full=self.SolsArray_Full.view(np.recarray)
        self.ListKapa=[[] for iAnt in range(na)]
        self.ListKeepKapa=[[] for iAnt in range(na)]
        self.ListStd=[[] for iAnt in range(na)]
        self.ListMax=[[] for iAnt in range(na)]


        self.G=NpShared.ToShared("%sSharedGains"%self.IdSharedMem,self.G)
        self.G0Iter=NpShared.ToShared("%sSharedGains0Iter"%self.IdSharedMem,self.G.copy())
        self.InitCovariance()

    def InitCovariance(self,FromG=False,sigP=0.1,sigQ=0.01):
        if self.SolverType!="KAFCA": return
        if self.Q!=None: return
        na=self.VS.MS.na
        nd=self.SM.NDir

        
        _,_,npol,_=self.G.shape
        

        if self.PolMode=="IDiag":
            npolx=2
            npoly=1
        elif self.PolMode=="Scalar":
            npolx=1
            npoly=1
        elif self.PolMode=="IFull":
            npolx=2
            npoly=2

        if FromG==False:
            P=(sigP**2)*np.array([np.diag(np.ones((nd*npolx*npoly,),self.DType)) for iAnt in range(na)])
            Q=(sigQ**2)*np.array([np.diag(np.ones((nd*npolx*npoly,),self.DType)) for iAnt in range(na)])
        else:

            P=(sigP**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*np.diag(np.ones((nd*npolx*npoly),self.DType)) for iAnt in range(na)])
            Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*np.diag(np.ones((nd*npolx*npoly),self.DType)) for iAnt in range(na)])


        if True:
            ra=self.SM.ClusterCat.ra
            dec=self.SM.ClusterCat.dec
            ns=ra.size
            
            d=np.sqrt((ra.reshape((ns,1))-ra.reshape((1,ns)))**2+(dec.reshape((ns,1))-dec.reshape((1,ns)))**2)
            d0=1e-5*np.pi/180
            QQ=(1./(1.+d/d0))**2
            Qa=np.zeros((nd,npolx,npoly,nd,npolx,npoly),self.DType)
            for ipol in range(npolx):
                for jpol in range(npoly):
                    Qa[:,ipol,jpol,:,ipol,jpol]=QQ[:,:]

            #Qa=np.zeros((nd,npolx,npoly,nd,npolx,npoly),self.DType)
            F=self.SM.ClusterCat.SumI.copy()
            F/=F.max()

            #stop
            if self.GD["Beam"]["BeamModel"]!=None:
                from killMS2.Data import ClassBeam
                BeamMachine=ClassBeam.ClassBeam(self.VS.MSName,self.GD,self.SM)
                AbsMeanBeam=BeamMachine.GiveMeanBeam()
                AbsMeanBeamAnt=np.mean(AbsMeanBeam[:,:,0,0,0],axis=1)
                for idir in range(nd):
                    Qa[idir,:,:,idir,:,:]*=(AbsMeanBeamAnt[idir]*F[idir])**2
                    #Qa[idir,:,:,idir,:,:]*=(F[idir])**2
            else:
                for idir in range(nd):
                    Qa[idir,:,:,idir,:,:]*=(F[idir])**2


    
            Qa=Qa.reshape((nd*npolx*npoly,nd*npolx*npoly))
            #print np.diag(Qa)
            Q=(sigQ**2)*np.array([np.max(np.abs(self.G[iAnt]))**2*Qa for iAnt in range(na)])

        self.P=P
        self.evP=np.zeros_like(P)
        self.P=NpShared.ToShared("%sSharedCovariance"%self.IdSharedMem,self.P)
        self.Q=NpShared.ToShared("%sSharedCovariance_Q"%self.IdSharedMem,Q)
        self.Q_Init=self.Q.copy()
        self.evP=NpShared.ToShared("%sSharedEvolveCovariance"%self.IdSharedMem,self.evP)
        nbuff=10

    def setNextData(self):
        DATA=self.VS.GiveNextVis()

        NDone,nt=self.pBarProgress
        intPercent=int(100*  NDone / float(nt))
        self.pBAR.render(intPercent, '%4i/%i' % (NDone,nt))

        if DATA=="EndOfObservation":
            print>>log, ModColor.Str("Reached end of data")
            return "EndOfObservation"
        if DATA=="EndChunk":
            print>>log, ModColor.Str("Reached end of data chunk")
            return "EndChunk"
        if DATA=="AllFlaggedThisTime":
            #print "AllFlaggedThisTime"
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
            #print>>log," rmsFromDataJacobAnt: %s"%self.rms
        elif self.rmsFromExt!=None:
            self.rms=self.rmsFromExt
            #print>>log," rmsFromExt: %s"%self.rms
        elif (self.TypeRMS=="GlobalData"):
            Dpol=DATA["data"][:,:,1:3]
            Fpol=DATA["flags"][:,:,1:3]
            self.rms=np.std(Dpol[Fpol==0])/np.sqrt(2.)
            #print>>log," rmsFromGlobalData: %s"%self.rms
        else:
            stop


        #print "rms=",self.rms

        return True

    def SetRmsFromExt(self,rms):
        self.rmsFromExt=rms

    def InitPlotGraph(self):
        from Plot import Graph
        print>>log,"Initialising plots ..." 
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

    def doNextTimeSolve(self):



        if self.G==None:
            self.InitSol()

        ListAntSolve=[i for i in range(self.VS.MS.na) if not(i in self.VS.FlagAntNumber)]
        self.DicoJM={}

        self.pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Solving ", HeaderSize=10,TitleSize=13)
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
        while True:
            self.pBarProgress=NDone,float(nt)
            NDone+=1
            T.reinit()

            print
            print "zeros=",np.count_nonzero(NpShared.GiveArray("%sPredictedData"%self.IdSharedMem))
            print
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
                JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,Precision="S",IdSharedMem=self.IdSharedMem,GD=self.GD,
                                        **self.ConfigJacobianAntenna)
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

                        x,_,_=JM.doLMStep(self.G)
                        if i==self.NIter-1: JM.PredictOrigFormat(self.G)
                    if self.SolverType=="KAFCA":
                        EM=ClassModelEvolution(iAnt,
                                               StepStart=3,
                                               WeigthScale=0.3,
                                               DoEvolve=True,
                                               BufferNPoints=10,
                                               sigQ=0.01,IdSharedMem=self.IdSharedMem)

                        x,P,_=JM.doEKFStep(self.G,self.P,self.evP,self.rms)
                        JM.PredictOrigFormat(self.G)
                        
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
        
        
        T=ClassTimeIt.ClassTimeIt("ClassWirtinger")
        T.disable()
                    


        #T=ClassTimeIt.ClassTimeIt()
        #T.disable()
        for ii in range(NCPU):
             
            W=WorkerAntennaLM(work_queue, result_queue,self.SM,self.PolMode,self.SolverType,self.IdSharedMem,
                              ConfigJacobianAntenna=self.ConfigJacobianAntenna,GD=self.GD)#,args=(e,))
            workerlist.append(W)
            workerlist[ii].start()



        ##############################

        T0,T1=self.VS.TimeMemChunkRange_sec[0],self.VS.TimeMemChunkRange_sec[1]
        DT=(T1-T0)
        dt=self.VS.TVisSizeMin*60.
        nt=int(DT/float(dt))+1
        

        self.pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Solving ", HeaderSize=10,TitleSize=13)
        if not(self.DoPBar): self.pBAR.disable()
        #pBAR.disable()
        self.pBAR.render(0, '%4i/%i' % (0,nt))
        NDone=0
        iiCount=0
        while True:
            T.reinit()
            self.pBarProgress=NDone,float(nt)
            Res=self.setNextData()
            NDone+=1
            T.timeit("read data")
            if Res=="EndChunk": break
            if Res=="AllFlaggedThisTime": continue
            #print "saving"
            #print "saving"
            #sols=self.GiveSols()
            #np.save("lastSols",sols)
            #print "done"
            if SkipMode:
                print iiCount
                iiCount+=1
                if iiCount<200: continue


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


            Gold=self.G.copy()
            DoCalcEvP=False
            if (self.CounterEvolveP())&(self.SolverType=="KAFCA")&(self.iCurrentSol>self.EvolvePStepStart):
                DoCalcEvP=True
            elif (self.SolverType=="KAFCA")&(self.iCurrentSol<=self.EvolvePStepStart):
                DoCalcEvP=True

            T.timeit("before iterloop")
            for LMIter in range(NIter):
                #print
                # for EKF

                #print "===================================================="
                #print "===================================================="
                #print "===================================================="
                #########
                if LMIter>0:
                    DoCalcEvP=False
                    DoEvP=False
                elif LMIter==0:
                    self.G0Iter[:]=self.G[:]
                    DoEvP=False

                if LMIter==(NIter-1):
                    DoEvP=True
                
                DoFullPredict=False
                if LMIter==(NIter-1):
                    DoFullPredict=True
                    
                u,v,w=self.DATA["uvw"].T
                A0=self.DATA["A0"]
                A1=self.DATA["A1"]
                meanW=np.zeros((self.VS.MS.na,),np.float32)
                for iAntMS in ListAntSolve:
                    ind=np.where((A0==iAntMS)|(A1==iAntMS))[0]
                    meanW[iAntMS]=np.mean(np.abs(w[ind]))
                meanW=meanW[ListAntSolve]
                indOrderW=np.argsort(meanW)[::-1]
                SortedWListAntSolve=(np.array(ListAntSolve)[indOrderW]).tolist()
                #print indOrderW

                for iAnt in SortedWListAntSolve:
                    work_queue.put((iAnt,DoCalcEvP,tm,self.rms,DoEvP,DoFullPredict))
 
                T.timeit("put in queue")
                rmsFromDataList=[]
                DTs=np.zeros((self.VS.MS.na,),np.float32)
                while iResult < NJobs:
                    iAnt,G,P,rmsFromData,InfoNoise,DT = result_queue.get()
                    if rmsFromData!=None:
                        rmsFromDataList.append(rmsFromData)
                    
                    #T.timeit("result_queue.get()")
                    self.G[iAnt][:]=G[:]
                    if type(P)!=type(None):
                        self.P[iAnt,:]=P[:]
                    
                    DTs[iAnt]=DT
                    kapa=InfoNoise["kapa"]
                    self.ListStd[iAnt].append(InfoNoise["std"])
                    self.ListMax[iAnt].append(InfoNoise["max"])
                    self.ListKeepKapa[iAnt].append(InfoNoise["kapa"])
                    self.SolsArray_Stats[self.iCurrentSol][iAnt][0]=InfoNoise["std"]
                    self.SolsArray_Stats[self.iCurrentSol][iAnt][1]=InfoNoise["max"]
                    self.SolsArray_Stats[self.iCurrentSol][iAnt][2]=InfoNoise["kapa"]
                    self.SolsArray_Stats[self.iCurrentSol][iAnt][3]=self.rms
                    
                    iResult+=1
                    if (kapa!=None)&(LMIter==0):
                        if kapa==-1.:
                            if len(self.ListKapa[iAnt])>0:
                                kapa=self.ListKapa[iAnt][-1]
                            else:
                                kapa=1.

                        self.ListKapa[iAnt].append(kapa)
                        dt=.5
                        TraceResidList=self.ListKapa[iAnt]
                        x=np.arange(len(TraceResidList))
                        expW=np.exp(-x/dt)[::-1]
                        expW/=np.sum(expW)
                        kapaW=np.sum(expW*np.array(TraceResidList))
                        #self.Q[iAnt]=(kapaW**2)*self.Q_Init[iAnt]

                        self.Q[iAnt][:]=(kapaW)*self.Q_Init[iAnt][:]

                        # self.Q[iAnt][:]=(kapaW)**2*self.Q_Init[iAnt][:]*1e6
                        # QQ=NpShared.FromShared("%sSharedCovariance_Q"%self.IdSharedMem)[iAnt]
                        # print self.Q[iAnt]-QQ[iAnt]
                        
                        #self.Q[iAnt]=self.Q_Init[iAnt]

                        #print iAnt,kapa,kapaW
                        #sig=np.sqrt(np.abs(np.array([np.diag(self.P[i]) for i in [iAnt]]))).flatten()
                        #print sig
                        

                T.timeit("getResult")
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
                    AntPlot=np.arange(self.VS.MS.na)#np.array(ListAntSolve)
                    pylab.clf()
                    pylab.plot(np.abs(self.G[AntPlot].flatten()))
                    pylab.plot(np.abs(Gold[AntPlot].flatten()))
                    
                    if self.SolverType=="KAFCA":

                        sig=[]
                        for iiAnt in AntPlot:
                            if iiAnt in ListAntSolve:
                                sig.append(np.sqrt(np.abs(np.array([np.diag(self.P[iiAnt]) ]))).flatten())
                            else:
                                sig.append(np.zeros((self.SM.NDir,),self.P.dtype))
                        
                        sig=np.array(sig).flatten()

                        pylab.plot(np.abs(self.G[AntPlot].flatten())+sig,color="black",ls="--")
                        pylab.plot(np.abs(self.G[AntPlot].flatten())-sig,color="black",ls="--")

                    pylab.ylim(0,2)
                    pylab.draw()
                    pylab.show(False)
                    pylab.pause(0.1)


                T.timeit("Plot")




                
            

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
            
            if OnlyOne: break


 
        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

            
        return True


    def AppendGToSolArray(self):
        t0,t1=self.VS.CurrentVisTimes_MS_Sec
        self.SolsArray_t0[self.iCurrentSol]=t0
        self.SolsArray_t1[self.iCurrentSol]=t1
        tm=(t0+t1)/2.
        self.SolsArray_tm[self.iCurrentSol]=tm
        self.SolsArray_done[self.iCurrentSol]=1
        self.SolsArray_G[self.iCurrentSol][:]=self.G[:]

 






#======================================
import multiprocessing
from killMS2.Predict.PredictGaussPoints_NumExpr5 import ClassPredict
class WorkerAntennaLM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,SM,PolMode,SolverType,IdSharedMem,ConfigJacobianAntenna=None,GD=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM
        self.PolMode=PolMode
        self.SolverType=SolverType
        self.IdSharedMem=IdSharedMem
        self.ConfigJacobianAntenna=ConfigJacobianAntenna
        self.GD=GD

        self.InitPM()

        #self.DoCalcEvP=DoCalcEvP
        #self.ThisTime=ThisTime
        #self.e,=kwargs["args"]
        

    def InitPM(self):

        x=np.linspace(0.,15,100000)
        Exp=np.float32(np.exp(-x))
        LExp=[Exp,x[1]-x[0]]
        
        self.PM=ClassPredict(Precision="S",DoSmearing=self.GD["SkyModel"]["Decorrelation"],IdMemShared=self.IdSharedMem,LExp=LExp)

        if self.GD["ImageSkyModel"]["BaseImageName"]!="":
            self.PM.InitGM(self.SM)

    def shutdown(self):
        self.exit.set()
    def run(self):

        while not self.kill_received:
            try:
                iAnt,DoCalcEvP,ThisTime,rms,DoEvP,DoFullPredict = self.work_queue.get()
            except:
                break
            #self.e.wait()
            
            T0=time.time()
            T=ClassTimeIt.ClassTimeIt("Worker Ant=%2.2i"%iAnt)
            T.disable()
            # if DoCalcEvP:
            #     T.disable()
            JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,PM=self.PM,IdSharedMem=self.IdSharedMem,GD=self.GD,
                                    **dict(self.ConfigJacobianAntenna))
            T.timeit("ClassJacobianAntenna")
            JM.setDATA_Shared()
            T.timeit("setDATA_Shared")

            G=NpShared.GiveArray("%sSharedGains"%self.IdSharedMem)
            G0Iter=NpShared.GiveArray("%sSharedGains0Iter"%self.IdSharedMem)
            P=NpShared.GiveArray("%sSharedCovariance"%self.IdSharedMem)
            #Q=NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem)
            evP=NpShared.GiveArray("%sSharedEvolveCovariance"%self.IdSharedMem)
            T.timeit("GiveArray")

            if self.SolverType=="CohJones":
                x,_,InfoNoise=JM.doLMStep(G)
                if DoFullPredict: JM.PredictOrigFormat(G)
                self.result_queue.put([iAnt,x,None,None,InfoNoise,0.])

            elif self.SolverType=="KAFCA":
                #T.disable()
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

                x,Pout,InfoNoise=JM.doEKFStep(G,P,evP,rms,Gains0Iter=G0Iter)
                T.timeit("EKFStep")
                if DoFullPredict: JM.PredictOrigFormat(G)
                T.timeit("PredictOrigFormat")
                rmsFromData=JM.rmsFromData

                if DoEvP:
                    Pa=EM.Evolve0(x,Pout)#,kapa=kapa)
                    T.timeit("Evolve")
                else:
                    Pa=P[iAnt].copy()
                #_,Pa=EM.Evolve(x,Pout,ThisTime)

                if type(Pa)!=type(None):
                    Pout=Pa

                DT=time.time()-T0
                self.result_queue.put([iAnt,x,Pout,rmsFromData,InfoNoise,DT])
