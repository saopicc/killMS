#!/usr/bin/env python

import optparse
import sys
from Other import MyPickle
from Other import logo
from Other import ModColor
from Other import MyLogger
log=MyLogger.getLogger("killMS")
MyLogger.itsLog.logger.setLevel(MyLogger.logging.CRITICAL)

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]

# test

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
from Sky import ClassSM
from Wirtinger.ClassWirtingerSolver import ClassWirtingerSolver

from Other import ClassTimeIt
from Data import ClassVisServer
#from Sky.PredictGaussPoints_NumExpr import ClassPredict
from Sky.PredictGaussPoints_NumExpr3 import ClassPredictParallel as ClassPredict 
from Array import ModLinAlg
from Array import NpShared
from Other import reformat

import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

def read_options():
    logo.print_logo()
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options")
    group.add_option('--ms',help='Input MS to draw [no default]',default='')
    group.add_option('--SkyModel',help='List of targets [no default]',default='')
    group.add_option('--TChunk',help=' Time Chunk in hours. Default is %default',default=15)
    group.add_option('--InCol',help=' Column to work on. Default is %default',default="CORRECTED_DATA_BACKUP")
    group.add_option('--OutCol',help=' Column to write to. Default is %default',default="CORRECTED_DATA")
    group.add_option('--LOFARBeam',help='(Mode, Time): Mode can be AE, E, or A for "Array factor" and "Element beam". Time is the estimation time step',default="")
    group.add_option('--UVMinMax',help=' Baseline length selection in km. For example --UVMinMax=0.1,100 selects baseline with length between 100 m and 100 km. Default is %default',default=None)
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Source selection options")
    group.add_option('--kills',help='Name or number index of sources to kill',default="")
    group.add_option('--invert',help='Invert the selected sources to kill',default="0")
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Solution options")
    group.add_option('--SubOnly',help=' Substract the skymodel assuming unity Jones matrices (no solve). Default is %default',default="0")
    group.add_option('--DoPlot',type="int",help=' Plot the solutions, for debugging. Default is %default',default=0)
    group.add_option('--DoSub',type="int",help=' Substact selected sources. Default is %default',default=1)
    group.add_option('--ApplyCal',help=' Apply direction averaged gains to residual data in the mentioned direction. \
    If ApplyCal=-1 takes the mean gain over directions. Default is %default',default="No")
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Algorithm options", "Default values should give reasonable results, but all of them have noticeable influence on the results")
    group.add_option('--SolverType',help='Name of the solver to use (CohJones/KAFCA)',default="CohJones")
    group.add_option('--NCPU',type="int",help=' Number of cores to use. Default is %default ',default=NCPU_default)
    group.add_option('--PolMode',help=' Polarisation mode (Scalar/HalfFull). Default is %default',default="Scalar")
    group.add_option('--dt',type="float",help='Time interval for a solution [minutes]. Default is %default. ',default=30)
    group.add_option('--ClearSHM',help='Clear shared memory with given ID [no default]',default='')
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* CohJones additional options")
    group.add_option('--NIter',type="int",help=' Number of iterations for the solve. Default is %default ',default=7)
    group.add_option('--Lambda',type="float",help=' Lambda parameter. Default is %default ',default=1)
    group.add_option('--DoBar',help=' Draw progressbar. Default is %default',default="1")
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* KAFCA additional options")
    group.add_option('--InitLM',type="int",help='Initialise Kalman filter with Levenberg Maquardt. Default is %default',default=1)
    group.add_option('--InitLM_dt',type="float",help='Time interval in minutes. Default is %default',default=5)
    group.add_option('--CovP',type="float",help='Initial prior Covariance in fraction of the initial gain amplitude. Default is %default',default=0.1) 
    group.add_option('--CovQ',type="float",help='Intrinsic process Covariance in fraction of the initial gain amplitude. Default is %default',default=0.01) 
    group.add_option('--evP_Step',type="int",help='Start calculation evP every evP_Step after that step. Default is %default',default=0)
    group.add_option('--evP_StepStart',type="int",help='Calcule (I-KJ) matrix every evP_Step steps. Default is %default',default=1)

    opt.add_option_group(group)
    
    
    options, arguments = opt.parse_args()
    options.DoPlot=(options.DoPlot==1)
    f = open("last_killMS.obj","wb")
    pickle.dump(options,f)
    
def PrintOptions(options,IdSharedMem):
    print ModColor.Str(" killMS configuration")
    print "   - MS Name: %s"%ModColor.Str(options.ms,col="green")
    print "   - Reading %s, and writting to %s"%(ModColor.Str(options.InCol,col="green"),ModColor.Str(options.OutCol,col="green"))


    
    # print options.TChunk
    if options.LOFARBeam!="":
        mode,DT=options.LOFARBeam.split(",")
        print "   - LOFAR beam in %s mode with DT=%s"%(mode,DT)

    #print options.kills
    #print options.invert
    #print options.SubOnly
    
    print "   - Apply calibration: %s"%(options.ApplyCal)


    print "   - Algorithm %s in %s mode [%i CPU]"%(ModColor.Str(options.SolverType,col="green") ,ModColor.Str(options.PolMode,col="green"),options.NCPU)
    print "   - Solution time interval %5.2f min."%options.dt

    if options.SolverType=="CohJones":
        print "   - Number of iterations %i"%options.NIter
    if options.SolverType=="KAFCA":
        print "   - Covariance %5.1f of the initial gain amplitude"%float(options.CovP)
        if options.InitLM==1:
            print "   - Initialise using Levenberg-Maquardt with dt=%5.1f"%float(options.InitLM_dt)

    print "   - IdSharedMem %s"%(IdSharedMem)
    print

def main(options=None):
    

    if options==None:
        f = open("last_killMS.obj",'rb')
        options = pickle.load(f)
    

    #IdSharedMem=str(int(np.random.rand(1)[0]*100000))+"."
    global IdSharedMem
    IdSharedMem=str(int(os.getpid()))+"."
    PrintOptions(options,IdSharedMem)
    ApplyCal=(options.ApplyCal=="1")

    if options.ms=="":
        print "Give an MS name!"
        exit()
    if options.SkyModel=="":
        print "Give a Sky Model!"
        exit()
    if not(".npy" in options.SkyModel):
        print "Give a numpy sky model!"
        exit()

    ApplyCal=None
    if options.ApplyCal!="No":
        ApplyCal=int(options.ApplyCal)

    TChunk=float(options.TChunk)
    dt=float(options.dt)
    dtInit=float(options.InitLM_dt)
    NCPU=int(options.NCPU)
    SubOnly=(int(options.SubOnly)==1)
    invert=(options.invert=="1")
    options.InitLM=(int(options.InitLM)==1)


    
    if options.kills!="":
        kills=options.kills.split(",")
    else:
        invert=True
        kills=[]

    ######################################

    NpShared.DelAll(IdSharedMem)
    ReadColName  = options.InCol
    WriteColName = options.OutCol

    DicoSelectOptions= {}
    if options.UVMinMax!=None:
        sUVmin,sUVmax=options.UVMinMax
        UVmin,UVmax=float(sUVmin),float(sUVmax)
        DicoSelectOptions["UVRangeKm"]=UVmin,UVmax


    SM=ClassSM.ClassSM(options.SkyModel,
                       killdirs=kills,invert=invert)


    #SM.SourceCat.I*=1000**2
    VS=ClassVisServer.ClassVisServer(options.ms,ColName=ReadColName,
                                     TVisSizeMin=dt,
                                     DicoSelectOptions=DicoSelectOptions,
                                     TChunkSize=TChunk,IdSharedMem=IdSharedMem)
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


    Solver=ClassWirtingerSolver(VS,SM,PolMode=options.PolMode,
                                BeamProps=BeamProps,
                                NIter=options.NIter,NCPU=NCPU,
                                SolverType=options.SolverType,
                                evP_Step=options.evP_Step,evP_StepStart=options.evP_StepStart,
                                DoPlot=options.DoPlot,
                                Lambda=options.Lambda,IdSharedMem=IdSharedMem)
    Solver.InitSol(TestMode=False)
    PM=ClassPredict(NCPU=NCPU,IdMemShared=IdSharedMem)
    SM=Solver.SM


    if (options.InitLM) & (options.SolverType=="KAFCA"):

        rms,SolverInit_G=GiveNoise(options,
                                   DicoSelectOptions,
                                   IdSharedMem,
                                   SM,PM)

        Solver.InitSol(G=SolverInit_G,TestMode=False)
        Solver.InitCovariance(FromG=True,sigP=options.CovP,sigQ=options.CovQ)

        Solver.SetRmsFromExt(rms)



    DoSubstract=(options.DoSub==1)
    #print "!!!!!!!!!!!!!!"
    #Solver.SetRmsFromExt(100)
    # Solver.InitCovariance(FromG=True,sigP=options.CovP,sigQ=options.CovQ)


    while True:

        Load=VS.LoadNextVisChunk()
        if Load=="EndOfObservation":
            break

        Solver.doNextTimeSolve_Parallel()
        #Solver.doNextTimeSolve_Parallel(SkipMode=True)
        #Solver.doNextTimeSolve()

        # substract
        ind=np.where(SM.SourceCat.kill==1)[0]
        
        if (ind.size>0)&((DoSubstract)|(ApplyCal!=None)):
            Sols=Solver.GiveSols()
            Jones={}
            Jones["t0"]=Sols.t0
            Jones["t1"]=Sols.t1
            nt,na,nd,_,_=Sols.G.shape
            G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
            Jones["Beam"]=G
            Jones["BeamH"]=ModLinAlg.BatchH(G)
            Jones["ChanMap"]=np.zeros((VS.MS.NSPWChan,))
            
            if DoSubstract:
                print>>log, ModColor.Str("Substract sources ... ",col="green")
                SM.SelectSubCat(SM.SourceCat.kill==1)

                PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                Solver.VS.ThisDataChunk["data"]-=PredictData
                SM.RestoreCat()


            if ApplyCal!=None:
                print>>log, ModColor.Str("Apply calibration in direction: %i"%ApplyCal,col="green")
                PM.ApplyCal(Solver.VS.ThisDataChunk,Jones,ApplyCal)

            Solver.VS.MS.data=Solver.VS.ThisDataChunk["data"]
            Solver.VS.MS.flags_all=Solver.VS.ThisDataChunk["flags"]
            Solver.VS.MS.SaveVis(Col=WriteColName)
    
    FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.ms),options.SolverType)
    print>>log, "Save Solutions in file: %s"%FileName
    Sols=Solver.GiveSols()
    StationNames=np.array(Solver.VS.MS.StationNames)
    np.savez(FileName,Sols=Sols,StationNames=StationNames,SkyModel=SM.ClusterCat,ClusterCat=SM.ClusterCat)
    NpShared.DelAll(IdSharedMem)

    
def GiveNoise(options,DicoSelectOptions,IdSharedMem,SM,PM):
    print>>log, ModColor.Str("Initialising Kalman filter with Levenberg-Maquardt estimate")
    dtInit=float(options.InitLM_dt)
    VSInit=ClassVisServer.ClassVisServer(options.ms,ColName=options.InCol,
                                         TVisSizeMin=dtInit,
                                         DicoSelectOptions=DicoSelectOptions,
                                         TChunkSize=dtInit/60,IdSharedMem=IdSharedMem)
    
    VSInit.LoadNextVisChunk()
    SolverInit=ClassWirtingerSolver(VSInit,SM,PolMode=options.PolMode,
                                    NIter=options.NIter,NCPU=options.NCPU,
                                    SolverType="CohJones",
                                    DoPlot=options.DoPlot,
                                    DoPBar=False,IdSharedMem=IdSharedMem)
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
    read_options()
    f = open("last_killMS.obj",'rb')
    options = pickle.load(f)
    if options.DoBar=="0":
        from Other.progressbar import ProgressBar
        ProgressBar.silent=1


    if options.ClearSHM!="":
        NpShared.DelAll(options.ClearSHM)
    else:
        try:
            main(options=options)
        except:
            NpShared.DelAll(IdSharedMem)
