#!/usr/bin/env python

import optparse
import sys
from Other import MyPickle
from Other import logo
from Other import ModColor
from Other import MyLogger
from Other import MyPickle
from Other import PrintOptParse
from Parset import MyOptParse

log=MyLogger.getLogger("killMS")
MyLogger.itsLog.logger.setLevel(MyLogger.logging.CRITICAL)

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]
from pyrap.tables import table
# test
SaveFile="last_killMS.obj"

#import numpy
#print numpy.__file__
#import pyrap
#print pyrap.__file__
#stop


if "nocol" in sys.argv:
    print "nocol"
    ModColor.silent=1
if "nox" in sys.argv:
    import matplotlib
    matplotlib.use('agg')
    print ModColor.Str(" == !NOX! ==")
    


import time
import os
import numpy as np
import pickle
from SkyModel.Sky import ClassSM
from Wirtinger.ClassWirtingerSolver import ClassWirtingerSolver

from Other import ClassTimeIt
from Data import ClassVisServer

from Predict.PredictGaussPoints_NumExpr4 import ClassPredictParallel as ClassPredict 
#from Predict.PredictGaussPoints_NumExpr2 import ClassPredictParallel as ClassPredict_orig
#from Predict.PredictGaussPoints_NumExpr4 import ClassPredict as ClassPredict 
#from Predict.PredictGaussPoints_NumExpr2 import ClassPredict as ClassPredict_orig
#from Sky.PredictGaussPoints_NumExpr4 import ClassPredict as ClassPredict 

#from Sky.PredictGaussPoints_NumExpr2 import ClassPredictParallel as ClassPredict_orig 
#from Sky.PredictGaussPoints_NumExpr3 import ClassPredict as ClassPredict 
#from Sky.PredictGaussPoints_NumExpr2 import ClassPredict as ClassPredict_orig 

from Array import ModLinAlg
from Array import NpShared
from Other import reformat

import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

from Parset import ReadCFG

global Parset
Parset=ReadCFG.Parset("%s/killMS2/Parset/DefaultParset.cfg"%os.environ["KILLMS_DIR"])


def read_options():
    D=Parset.DicoPars

    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""

    OP=MyOptParse.MyOptParse(usage='Usage: %prog --MSName=somename.MS --SkyModel=SM.npy <options>',description=desc,
                             DefaultDict=D)


    #opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',description=desc)
    OP.OptionGroup("* Data-related options","VisData")
    OP.add_option('MSName',help='Input MS to draw [no default]')
    OP.add_option('TChunk',help='Time Chunk in hours. Default is %default')
    OP.add_option('InCol',help='Column to work on. Default is %default')
    OP.add_option('OutCol',help='Column to write to. Default is %default')

    OP.OptionGroup("* Sky related options","SkyModel")
    OP.add_option('SkyModel',help='List of targets [no default]')
    OP.add_option('LOFARBeam',help='(Mode, Time): Mode can be AE, E, or A for "Array factor" and "Element beam". Time is the estimation time step')
    OP.add_option('kills',help='Name or number index of sources to kill')
    OP.add_option('invert',help='Invert the selected sources to kill')
    OP.add_option('Decorrelation',type="str",help=' . Default is %default')

    OP.OptionGroup("* Data Selection","DataSelection")
    OP.add_option('UVMinMax',help='Baseline length selection in km. For example UVMinMax=0.1,100 selects baseline with length between 100 m and 100 km. Default is %default')
    OP.add_option('FlagAnts',type="str",help='FlagAntenna patern. Default is %default')
    OP.add_option('DistMaxToCore',type="float",help='Maximum distance to core in km. Default is %default')

    OP.OptionGroup("* Weighting scheme","Weighting")
    OP.add_option('Resolution',type="float",help='Resolution in arcsec. Default is %default')
    OP.add_option('Weighting',type="str",help='Weighting scheme. Default is %default')
    OP.add_option('Robust',type="float",help='Briggs Robust parameter. Default is %default')
    
    OP.OptionGroup("* Action options","Actions")
    OP.add_option('DoPlot',type="int",help='Plot the solutions, for debugging. Default is %default')
    OP.add_option('SubOnly',type="int",help='Substact selected sources. Default is %default')
    OP.add_option('DoBar',help=' Draw progressbar. Default is %default',default="1")
    OP.add_option('NCPU',type="int",help='Number of cores to use. Default is %default ')
    # OP.add_option('ApplyCal',type="int",help='Apply direction averaged gains to residual data in the mentioned direction. \
    # If ApplyCal=-1 takes the mean gain over directions. -2 if off. Default is %default')

    OP.OptionGroup("* Solution-related options","Solutions")
    OP.add_option('ExtSols',type="str",help='External solution file. If set, will not solve.')
    #OP.add_option('ApplyMode',type="str",help='Substact selected sources. ')
    OP.add_option('ClipMethod',type="str",help='Clip data in the IMAGING_WEIGHT column. Can be set to Resid or DDEResid . Default is %default')

    OP.OptionGroup("* Solver options","Solvers")
    OP.add_option('SolverType',help='Name of the solver to use (CohJones/KAFCA)')
    OP.add_option('PolMode',help='Polarisation mode (Scalar/HalfFull). Default is %default')
    OP.add_option('dt',type="float",help='Time interval for a solution [minutes]. Default is %default. ')
    
    OP.OptionGroup("* CohJones additional options","CohJones")
    OP.add_option('NIter',type="int",help=' Number of iterations for the solve. Default is %default ',default=7)
    OP.add_option('Lambda',type="float",help=' Lambda parameter. Default is %default ',default=1)

    OP.OptionGroup("* KAFCA additional options","KAFCA")
    OP.add_option('InitLM',type="int",help='Initialise Kalman filter with Levenberg Maquardt. Default is %default',default=1)
    OP.add_option('InitLMdt',type="float",help='Time interval in minutes. Default is %default',default=5)
    OP.add_option('CovP',type="float",help='Initial prior Covariance in fraction of the initial gain amplitude. Default is %default',default=0.1) 
    OP.add_option('CovQ',type="float",help='Intrinsic process Covariance in fraction of the initial gain amplitude. Default is %default',default=0.01) 
    OP.add_option('evPStep',type="int",help='Start calculation evP every evP_Step after that step. Default is %default',default=0)
    OP.add_option('evPStepStart',type="int",help='Calcule (I-KJ) matrix every evP_Step steps. Default is %default',default=1)
    

    OP.Finalise()
    OP.ReadInput()
    options=OP.GiveOptionObject()
    if options.SolverType=="KAFCA":
        RejectGroup=["CohJones"]
    elif options.SolverType=="CohJones":
        RejectGroup=["KAFCA"]

    OP.Print(RejectGroup)

    
    # #optcomplete.autocomplete(opt)

    # options, arguments = opt.parse_args()
    MyPickle.Save(OP,SaveFile)
    return OP
    

def main(OP=None,MSName=None):
    

    if OP==None:
        OP = MyPickle.Load(SaveFile)

        
    options=OP.GiveOptionObject()

    #IdSharedMem=str(int(np.random.rand(1)[0]*100000))+"."
    global IdSharedMem
    IdSharedMem=str(int(os.getpid()))+"."
    DoApplyCal=0#(options.ApplyCal!=-2)
    ApplyCal=0#int(options.ApplyCal)
    ReWeight=(options.ClipMethod!="")

    if MSName!=None:
        options.MSName=MSName

    if options.MSName=="":
        print "Give an MS name!"
        exit()
    if options.SkyModel=="":
        print "Give a Sky Model!"
        exit()
    if not(".npy" in options.SkyModel):
        print "Give a numpy sky model!"
        exit()


    TChunk=float(options.TChunk)
    dt=float(options.dt)
    dtInit=float(options.InitLMdt)
    NCPU=int(options.NCPU)
    #SubOnly=(int(options.SubOnly)==1)

    invert=(options.invert==True)

    options.InitLM=(int(options.InitLM)==1)
    DoSmearing=options.Decorrelation
    
    if type(options.kills)==str:
        kills=options.kills.split(",")
    elif type(options.kills)==list:
        kills=options.kills
        

    ######################################

    NpShared.DelAll(IdSharedMem)
    ReadColName  = options.InCol
    WriteColName = options.OutCol

    DicoSelectOptions= {}
    if options.UVMinMax!=None:
        sUVmin,sUVmax=options.UVMinMax#.split(",")
        UVmin,UVmax=float(sUVmin),float(sUVmax)
        DicoSelectOptions["UVRangeKm"]=UVmin,UVmax
    if options.FlagAnts!="":
        FlagAnts=options.FlagAnts#.split(",")
        DicoSelectOptions["FlagAnts"]=FlagAnts

    DicoSelectOptions["DistMaxToCore"]=options.DistMaxToCore


    SM=ClassSM.ClassSM(options.SkyModel,
                       killdirs=kills,invert=invert)
    

    #SM.SourceCat.I*=1000**2
    VS=ClassVisServer.ClassVisServer(options.MSName,ColName=ReadColName,
                                     TVisSizeMin=dt,
                                     DicoSelectOptions=DicoSelectOptions,
                                     TChunkSize=TChunk,IdSharedMem=IdSharedMem,
                                     SM=SM,NCPU=NCPU,
                                     Weighting=options.Weighting,
                                     Robust=options.Robust)
    print VS.MS
    if not(WriteColName in VS.MS.ColNames):
        print>>log, "Column %s not in MS "%WriteColName
        exit()
    if not(ReadColName in VS.MS.ColNames):
        print>>log, "Column %s not in MS "%ReadColName
        exit()

    BeamProps=None
    if options.LOFARBeam!="":
        Mode,sTimeMin=options.LOFARBeam.split(",")
        TimeMin=float(sTimeMin)
        BeamProps=Mode,TimeMin

    ResolutionRad=(options.Resolution/3600)*(np.pi/180)
    ConfigJacobianAntenna={"DoSmearing":DoSmearing,
                           "ResolutionRad":ResolutionRad,
                           "Lambda":options.Lambda,
                           "DoReg":False,#True,
                           "gamma":1,
                           "AmpQx":.5}

    Solver=ClassWirtingerSolver(VS,SM,PolMode=options.PolMode,
                                BeamProps=BeamProps,
                                NIter=options.NIter,NCPU=NCPU,
                                SolverType=options.SolverType,
                                evP_Step=options.evPStep,evP_StepStart=options.evPStepStart,
                                DoPlot=options.DoPlot,
                                DoPBar=options.DoBar,
                                IdSharedMem=IdSharedMem,
                                ConfigJacobianAntenna=ConfigJacobianAntenna)
    Solver.InitSol(TestMode=False)

    PM=ClassPredict(NCPU=NCPU,IdMemShared=IdSharedMem,DoSmearing=DoSmearing)
    PM2=None#ClassPredict_orig(NCPU=NCPU,IdMemShared=IdSharedMem)



    if (options.SolverType=="KAFCA"):

        if (options.InitLM):
            rms,SolverInit_G=GiveNoise(options,
                                       DicoSelectOptions,
                                       IdSharedMem,
                                       SM,PM,PM2,ConfigJacobianAntenna)
            dtype=SolverInit_G.dtype
            SolverInit_G=np.array(np.abs(SolverInit_G),dtype=dtype)
            Solver.InitSol(G=SolverInit_G,TestMode=False)
            Solver.InitCovariance(FromG=True,sigP=options.CovP,sigQ=options.CovQ)
            
            Solver.SetRmsFromExt(rms)
        else:
            Solver.InitCovariance(sigP=options.CovP,sigQ=options.CovQ)
            pass
            #Solver.SetRmsFromExt(100)


    #DoSubstract=(options.DoSub==1)
    ind=np.where(SM.SourceCat.kill==1)[0]
    DoSubstract=(ind.size>0)
    #print "!!!!!!!!!!!!!!"
    #
    # Solver.InitCovariance(FromG=True,sigP=options.CovP,sigQ=options.CovQ)

    SaveSols=False
    while True:

        Load=VS.LoadNextVisChunk()
        if Load=="EndOfObservation":
            break

        if options.ExtSols=="":
            SaveSols=True
            Solver.doNextTimeSolve_Parallel()
            #Solver.doNextTimeSolve_Parallel(SkipMode=True)
            #Solver.doNextTimeSolve()
            Sols=Solver.GiveSols()
        else:
            Sols=np.load(options.ExtSols)["Sols"]
            Sols=Sols.view(np.recarray)

        # substract
        #ind=np.where(SM.SourceCat.kill==1)[0]
        if ((DoSubstract)|(DoApplyCal)|(ReWeight)):
            Sols.t1[-1]+=1e3
            Jones={}
            Jones["t0"]=Sols.t0
            Jones["t1"]=Sols.t1
            Jones["t1"]=Sols.t1
            nt,na,nd,_,_=Sols.G.shape
            G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
            G=np.require(G, dtype=np.complex64, requirements="C_CONTIGUOUS")

            # if not("A" in options.ApplyMode):
            #     gabs=np.abs(G)
            #     gabs[gabs==0]=1.
            #     G/=gabs


            Jones["Beam"]=G
            Jones["BeamH"]=ModLinAlg.BatchH(G)
            Jones["ChanMap"]=np.zeros((VS.MS.NSPWChan,))
            times=Solver.VS.ThisDataChunk["times"]

            # ind=np.array([],np.int32)
            # for it in range(nt):
            #     t0=Jones["t0"][it]
            #     t1=Jones["t1"][it]
            #     indMStime=np.where((times>=t0)&(times<t1))[0]
            #     indMStime=np.ones((indMStime.size,),np.int32)*it
            #     ind=np.concatenate((ind,indMStime))

            DicoJonesMatrices=Jones
            ind=np.zeros((times.size,),np.int32)
            nt,na,nd,_,_,_=G.shape
            ii=0
            for it in range(nt):
                t0=DicoJonesMatrices["t0"][it]
                t1=DicoJonesMatrices["t1"][it]
                indMStime=np.where((times>=t0)&(times<t1))[0]
                indMStime=np.ones((indMStime.size,),np.int32)*it
                ind[ii:ii+indMStime.size]=indMStime[:]
                ii+=indMStime.size




            Jones["MapJones"]=ind



            if ReWeight:
                print>>log, ModColor.Str("Clipping bad solution-based data ... ",col="green")
                

                nrows=Solver.VS.ThisDataChunk["times"].size

                Solver.VS.ThisDataChunk["W"]=np.ones((nrows,Solver.VS.MS.ChanFreq.size),np.float64)


                ################
                # #PM.GiveCovariance(Solver.VS.ThisDataChunk,Jones)

                print>>log,"   Compute residual data"
                Predict=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                
                Solver.VS.ThisDataChunk["resid"]=Solver.VS.ThisDataChunk["data"]-Predict
                Weights=Solver.VS.ThisDataChunk["W"]

                if "Resid" in options.ClipMethod:
                    Diff=Solver.VS.ThisDataChunk["resid"]
                    std=np.std(Diff[Solver.VS.ThisDataChunk["flags"]==0])
                    print>>log, "   Estimated standard deviation in the residual data: %f"%std

                    ThresHold=5.
                    cond=(np.abs(Diff)>ThresHold*std)
                    ind=np.any(cond,axis=2)
                    Weights[ind]=0.


                if "DDEResid" in options.ClipMethod:
                    print>>log,"   Compute corrected residual data in all direction"
                    PM.GiveCovariance(Solver.VS.ThisDataChunk,Jones,SM)

                Weights=Solver.VS.ThisDataChunk["W"]
                NNotFlagged=np.count_nonzero(Weights)
                print>>log,"   Set weights to Zero for %5.2f %% of data"%(100*float(Weights.size-NNotFlagged)/(Weights.size))

                # ################
                # T=ClassTimeIt.ClassTimeIt()
                
                # #PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                # #T.timeit("a")
                # PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                # #T.timeit("b")

                # ################


                # Weights=Solver.VS.ThisDataChunk["W"]
                # Weights=Weights.reshape((Weights.size,1))*np.ones((1,4))
                # Solver.VS.MS.Weights[:]=Weights[:]

                print>>log, "  Writting in IMAGING_WEIGHT column "
                t=table(Solver.VS.MS.MSName,readonly=False,ack=False)
                t.putcol("IMAGING_WEIGHT",Weights)
                t.close()

                
            if DoSubstract:
                print>>log, ModColor.Str("Substract sources ... ",col="green")
                SM.SelectSubCat(SM.SourceCat.kill==1)

                

                if options.SubOnly==1:
                    print>>log, ModColor.Str(" Sublonly ... ",col="green")
                    PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM)
                else:
                    #print "timemap:",Jones["MapJones"][1997:1999]
                    #print "Jt0d0a35",Jones["Beam"][0,0,35]
                    #print "Jt1d0a0",Jones["Beam"][1,0,0]
                    PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                    #PredictData2=PM2.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                    #diff=(PredictData-PredictData2)
                    #print diff
                    #ind=np.where(diff==np.max(diff))
                    #print ind
                    #print np.max(PredictData-PredictData2)
                    #print np.where(np.isnan(diff))
                    #print PredictData[1997:1999],PredictData[1997:1999]
                Solver.VS.ThisDataChunk["data"]-=PredictData
                SM.RestoreCat()


            # if DoApplyCal:
            #     print>>log, ModColor.Str("Apply calibration in direction: %i"%ApplyCal,col="green")
            #     PM.ApplyCal(Solver.VS.ThisDataChunk,Jones,ApplyCal)

            Solver.VS.MS.data=Solver.VS.ThisDataChunk["data"]
            Solver.VS.MS.flags_all=Solver.VS.ThisDataChunk["flags"]
            # Solver.VS.MS.SaveVis(Col=WriteColName)

            if (DoSubstract|DoApplyCal):
                print>>log, "Save visibilities in %s column"%WriteColName
                t=table(Solver.VS.MS.MSName,readonly=False,ack=False)
                t.putcol(WriteColName,Solver.VS.MS.data,Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.putcol("FLAG",Solver.VS.MS.flags_all,Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.close()

                


    if SaveSols:
        FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),options.SolverType)
        print>>log, "Save Solutions in file: %s"%FileName
        Sols=Solver.GiveSols()
        StationNames=np.array(Solver.VS.MS.StationNames)
        np.savez(FileName,Sols=Sols,StationNames=StationNames,SkyModel=SM.ClusterCat,ClusterCat=SM.ClusterCat)

    NpShared.DelAll(IdSharedMem)

    
def GiveNoise(options,DicoSelectOptions,IdSharedMem,SM,PM,PM2,ConfigJacobianAntenna):
    print>>log, ModColor.Str("Initialising Kalman filter with Levenberg-Maquardt estimate")
    dtInit=float(options.InitLMdt)
    VSInit=ClassVisServer.ClassVisServer(options.MSName,ColName=options.InCol,
                                         TVisSizeMin=dtInit,
                                         DicoSelectOptions=DicoSelectOptions,
                                         TChunkSize=dtInit/60,IdSharedMem=IdSharedMem,
                                         SM=SM,NCPU=options.NCPU)
    
    VSInit.LoadNextVisChunk()
    # # test
    # PredictData=PM.predictKernelPolCluster(VSInit.ThisDataChunk,SM)
    # PredictData2=PM2.predictKernelPolCluster(VSInit.ThisDataChunk,SM)
    # print np.max(PredictData-PredictData2)
    # stop
    # #######
    SolverInit=ClassWirtingerSolver(VSInit,SM,PolMode=options.PolMode,
                                    NIter=options.NIter,NCPU=options.NCPU,
                                    SolverType="CohJones",
                                    #DoPlot=options.DoPlot,
                                    DoPBar=False,IdSharedMem=IdSharedMem,
                                    ConfigJacobianAntenna=ConfigJacobianAntenna)
    SolverInit.InitSol(TestMode=False)
    SolverInit.doNextTimeSolve_Parallel(OnlyOne=True)
    #SolverInit.doNextTimeSolve()
    Sols=SolverInit.GiveSols()
    Jones={}
    Jones["t0"]=Sols.t0
    Jones["t1"]=Sols.t1
    nt,na,nd,_,_=Sols.G.shape
    G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
    Jones["Beam"]=G
    Jones["BeamH"]=ModLinAlg.BatchH(G)
    Jones["ChanMap"]=np.zeros((VSInit.MS.NSPWChan,))

    # ind=np.array([],np.int32)
    # for it in range(nt):
    #     t0=Jones["t0"][it]
    #     t1=Jones["t1"][it]
    #     indMStime=np.where((SolverInit.VS.ThisDataChunk["times"]>=t0)&(SolverInit.VS.ThisDataChunk["times"]<t1))[0]
    #     indMStime=np.ones((indMStime.size,),np.int32)*it
    #     ind=np.concatenate((ind,indMStime))

    times=SolverInit.VS.ThisDataChunk["times"]
    DicoJonesMatrices=Jones
    ind=np.zeros((times.size,),np.int32)
    nt,na,nd,_,_,_=G.shape
    ii=0
    for it in range(nt):
        t0=DicoJonesMatrices["t0"][it]
        t1=DicoJonesMatrices["t1"][it]
        indMStime=np.where((times>=t0)&(times<t1))[0]
        indMStime=np.ones((indMStime.size,),np.int32)*it
        ind[ii:ii+indMStime.size]=indMStime[:]
        ii+=indMStime.size


    Jones["MapJones"]=ind
    PredictData=PM.predictKernelPolCluster(SolverInit.VS.ThisDataChunk,SolverInit.SM,ApplyTimeJones=Jones)

    SolverInit.VS.ThisDataChunk["data"]-=PredictData


    PM.ApplyCal(SolverInit.VS.ThisDataChunk,Jones,0)
    DATA=SolverInit.VS.ThisDataChunk
    A0=SolverInit.VS.ThisDataChunk["A0"]
    A1=SolverInit.VS.ThisDataChunk["A1"]
    _,nchan,_=DATA["data"].shape
    na=VSInit.MS.na
    rmsAnt=np.zeros((na,nchan,4),float)
    for A in range(na):
        ind=np.where((A0==1)|(A1==A))[0]
        Dpol=DATA["data"][ind,:,:]
        Fpol=DATA["flags"][ind,:,:]
        _,nchan,_=Dpol.shape
        # print "Antenna-%i"%A
        for ichan in range(nchan):
            
            d=Dpol[:,ichan,:]
            f=Fpol[:,ichan,:]
            # print 
            for ipol in range(4):
                dp=d[:,ipol]
                fp=f[:,ipol]
                rms=np.std(dp[fp==0])/np.sqrt(2.)
                mean=np.mean(dp[fp==0])/np.sqrt(2.)
                #print "    pol=%i: (mean, rms)=(%s, %s)"%(ipol, str(mean),str(rms))
                rmsAnt[A,ichan,ipol]=rms

    rmsAnt=np.mean(np.mean(rmsAnt[:,:,1:3],axis=2),axis=1)
    Mean_rmsAnt=np.mean(rmsAnt)
    Thr=5
    indFlag=np.where((rmsAnt-Mean_rmsAnt)/Mean_rmsAnt>Thr)[0]
    if indFlag.size>0:
        Stations=np.array(SolverInit.VS.MS.StationNames)
        print>>log, "Antenna %s have abnormal noise (Numbers %s)"%(str(Stations[indFlag]),str(indFlag))
    
    indTake=np.where((rmsAnt-Mean_rmsAnt)/Mean_rmsAnt<Thr)[0]
     
    gscale=np.mean(np.abs(G[:,:,indTake,:,0,0]))
    TrueMeanRMSAnt=np.mean(rmsAnt[indTake])
      
    GG=np.mean(np.mean(np.mean(np.abs(G[0,:]),axis=0),axis=0),axis=0)
    GGprod= np.dot( np.dot(GG,np.ones((2,2),float)*TrueMeanRMSAnt) , GG.T)
    rms=np.mean(GGprod)
    print>>log, "Estimated rms: %f Jy"%(rms)
    return rms,SolverInit.G



if __name__=="__main__":
    os.system('clear')
    logo.print_logo()


    ParsetFile=sys.argv[1]

    TestParset=ReadCFG.Parset(ParsetFile)
    if TestParset.Success==True:
        #global Parset
        Parset=TestParset
        print >>log,ModColor.Str("Successfully read %s parset"%ParsetFile)

    OP=read_options()
    options=OP.GiveOptionObject()

    if options.DoBar=="0":
        from Other.progressbar import ProgressBar
        ProgressBar.silent=1

    
    #main(OP=OP)

    import glob
    if "*" in options.MSName:
        Patern=options.MSName
        lMS=sorted(glob.glob(Patern))
        print>>log, "In batch mode, running killMS on the following MS:"
        for MS in lMS:
            print>>log, "  %s"%MS
    else:
        lMS=[options.MSName]

    
 
    try:
        for MSName in lMS:
            main(OP=OP,MSName=MSName)
    except:
        NpShared.DelAll(IdSharedMem)
            
