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
from Sky.PredictGaussPoints_NumExpr import ClassPredict
from Array import ModLinAlg
from Array import NpShared

import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

def read_options():
    logo.print_logo()
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--ms',help='Input MS to draw [no default]',default='')
    group.add_option('--SkyModel',help='List of targets [no default]',default='')
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Visibilities options")
    group.add_option('--TChunk',help=' Time Chunk in hours. Default is %default',default=15)
    group.add_option('--InCol',help=' Column to work on. Default is %default',default="CORRECTED_DATA_BACKUP")
    group.add_option('--OutCol',help=' Column to write to. Default is %default',default="CORRECTED_DATA")
    group.add_option('--LOFARBeam',help='(Mode, Time): Mode can be AE, E, or A for "Array factor" and "Element beam". Time is the estimation time step',default="")
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Source selection options")
    group.add_option('--kills',help='Name or number index of sources to kill',default="")
    group.add_option('--invert',help='Invert the selected sources to kill',default="0")
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Solution options")
    group.add_option('--SubOnly',help=' Only substract the skymodel. Default is %default',default="0")
    group.add_option('--ApplyCal',help=' Apply direction averaged gains to residual data in the mentioned direction. \
    If ApplyCal=-1 takes the mean gain over directions. Default is %default',default="No")
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Algorithm options", "Default values should give reasonable results, but all of them have noticeable influence on the results")
    group.add_option('--SolverType',help='Name of the solver to use (CohJones/KAFCA)',default="CohJones")
    group.add_option('--NCPU',type="int",help=' Number of cores to use. Default is %default ',default=NCPU_default)
    group.add_option('--PolMode',help=' Polarisation mode (Scalar/HalfFull). Default is %default',default="Scalar")
    group.add_option('--dt',type="float",help='Time interval for a solution [minutes]. Default is %default. ',default=30)
    group.add_option('--niter',type="int",help=' Number of iterations for the solve. Default is %default ',default=20)
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* KAFCA additional options")
    group.add_option('--InitLM',type="int",help='Initialise Kalman filter with Levenberg Maquardt. Default is %default',default=1)
    group.add_option('--InitLM_dt',type="float",help='Time interval in minutes. Default is %default',default=5)
    group.add_option('--CovP',type="float",help='Initial Covariance in fraction of the gain amplitude. Default is %default',default=0.1)
    opt.add_option_group(group)
    
    
    options, arguments = opt.parse_args()
    
    f = open("last_killMS.obj","wb")
    pickle.dump(options,f)
    
def PrintOptions(options):
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
        print "   - Number of iterations %i"%options.niter
    if options.SolverType=="KAFCA":
        print "   - Covariance %5.1f of the initial gain amplitude"%float(options.CovP)
        if options.InitLM==1:
            print "   - Initialise using Levenberg-Maquardt with dt=%5.1f"%float(options.InitLM_dt)

    print

def main(options=None):
    

    if options==None:
        f = open("last_killMS.obj",'rb')
        options = pickle.load(f)
    

    PrintOptions(options)
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
    niterin=int(options.niter)
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

    NpShared.DelAll()
    ReadColName  = options.InCol
    WriteColName = options.OutCol

    SM=ClassSM.ClassSM(options.SkyModel,
                       killdirs=kills,invert=invert)
    
    VS=ClassVisServer.ClassVisServer(options.ms,ColName=ReadColName,
                                     TVisSizeMin=dt,
                                     TChunkSize=TChunk)
    print VS.MS
    VS.LoadNextVisChunk()
    BeamProps=None
    if options.LOFARBeam!="":
        Mode,sTimeMin=options.LOFARBeam.split(",")
        TimeMin=float(sTimeMin)
        BeamProps=Mode,TimeMin

    Solver=ClassWirtingerSolver(VS,SM,PolMode=options.PolMode,
                                BeamProps=BeamProps,
                                NIter=niterin,NCPU=NCPU,
                                SolverType=options.SolverType)
    Solver.InitSol(TestMode=False)
    PM=ClassPredict()
    SM=Solver.SM


    if (options.InitLM) & (options.SolverType=="KAFCA"):
        
        print>>log, ModColor.Str("Initialising Kalman filter with Levenberg-Maquardt estimate")
        VSInit=ClassVisServer.ClassVisServer(options.ms,ColName=ReadColName,
                                             TVisSizeMin=dtInit,
                                             TChunkSize=TChunk)
        
        VSInit.LoadNextVisChunk()
        SolverInit=ClassWirtingerSolver(VSInit,SM,PolMode=options.PolMode,
                                        NIter=niterin,NCPU=NCPU,
                                        SolverType="CohJones")
        Res=SolverInit.setNextData()
        SolverInit.InitSol(TestMode=False)
        SolverInit.doNextTimeSolve_Parallel()
        Solver.InitSol(G=SolverInit.G,TestMode=False)
        Solver.InitCovariance(FromG=True,sigP=options.CovP)




    while True:
        Res=Solver.setNextData()

        if Res==True:
            # if not(SubOnly):
            #     Solver.doNextTimeSolve_Parallel()

            Solver.doNextTimeSolve_Parallel()

            # Solver.doNextTimeSolve()
            continue
        else:
            # substract
            
            Sols=Solver.GiveSols()
            Jones={}
            Jones["t0"]=Sols.t0
            Jones["t1"]=Sols.t1
            nt,na,nd,_,_=Sols.G.shape
            G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
            Jones["Beam"]=G
            Jones["BeamH"]=ModLinAlg.BatchH(G)
            Jones["ChanMap"]=np.zeros((VS.MS.NSPWChan,)).tolist()

            SM.SelectSubCat(SM.SourceCat.kill==1)
            PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
            SM.RestoreCat()


            # import pylab
            # pylab.clf()
            # nbl=1#Solver.VS.MS.nbl
            # a=Solver.VS.ThisDataChunk["data"][1::nbl,:,:].flatten().real
            # b=PredictData[1::nbl,:,:].flatten().real
            # pylab.plot(a)
            # pylab.plot(b)
            # pylab.plot(a-b)
            # pylab.draw()
            # pylab.show(False)
            # stop

            Solver.VS.ThisDataChunk["data"]-=PredictData

            if ApplyCal!=None:
                PM.ApplyCal(Solver.VS.ThisDataChunk,Jones,ApplyCal)


            Solver.VS.MS.data=Solver.VS.ThisDataChunk["data"]

            

            Solver.VS.MS.SaveVis(Col=WriteColName)

        if Res=="EndChunk":
            Load=VS.LoadNextVisChunk()
            if Load=="EndOfObservation":
                break


    NpShared.DelAll()

    




if __name__=="__main__":
    read_options()
    f = open("last_killMS.obj",'rb')
    options = pickle.load(f)

    main(options=options)
