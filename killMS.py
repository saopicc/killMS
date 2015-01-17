#!/usr/bin/env python

import optparse
import sys
from Other import MyPickle
from Other import logo
from Other import ModColor

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
from Other import MyLogger
log=MyLogger.getLogger("killMS")
import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

def read_options():
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--ms',help='Input MS to draw [no default]',default='')
    group.add_option('--SkyModel',help='List of targets [no default]',default='')
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Visibilities options")
    group.add_option('--TChunk',help=' Time Chunk in hours. Default is %default',default="15")
    group.add_option('--InCol',help=' Column to work on. Default is %default',default="CORRECTED_DATA_BACKUP")
    group.add_option('--OutCol',help=' Column to write to. Default is %default',default="CORRECTED_DATA")
    group.add_option('--LOFARBeamParms',help='Not Working yet',default="")
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
    group.add_option('--NCPU',help=' Number of cores to use. Default is %default ',default=NCPU_default)
    group.add_option('--PolMode',help=' Polarisation mode (Scalar/HalfFull). Default is %default',default="Scalar")
    group.add_option('--dt',help='Time interval for a solution [minutes]. Default is %default. ',default=30)
    group.add_option('--niter',help=' Number of iterations for the solve. Default is %default ',default="20")
    opt.add_option_group(group)
    
    
    options, arguments = opt.parse_args()
    
    f = open("last_killMS.obj","wb")
    pickle.dump(options,f)
    

def main(options=None):
    

    if options==None:
        f = open("last_killMS.obj",'rb')
        options = pickle.load(f)
    
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
    delta_time=float(options.dt)
    niterin=int(options.niter)
    NCPU=int(options.NCPU)
    SubOnly=(int(options.SubOnly)==1)
    invert=(options.invert=="1")
    
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
                                     TVisSizeMin=delta_time,
                                     TChunkSize=TChunk)


    

    # LM=ClassLM(VS,SM,PolMode="HalfFull")
    LM=ClassWirtingerSolver(VS,SM,PolMode=options.PolMode,
                            NIter=niterin,NCPU=NCPU,
                            SolverType=options.SolverType)
    LM.InitSol(TestMode=False)
    PM=ClassPredict()
    SM=LM.SM

    while True:
        Res=LM.setNextData()

        if Res==True:
            # if not(SubOnly):
            #     LM.doNextTimeSolve_Parallel()
            LM.doNextTimeSolve_Parallel()

            # LM.doNextTimeSolve()
            continue
        else:
            # substract
            
            Sols=LM.GiveSols()
            Jones={}
            Jones["t0"]=Sols.t0
            Jones["t1"]=Sols.t1
            nt,na,nd,_,_=Sols.G.shape
            G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
            Jones["Beam"]=G
            Jones["BeamH"]=ModLinAlg.BatchH(G)
            Jones["ChanMap"]=np.zeros((VS.MS.NSPWChan,)).tolist()

            SM.SelectSubCat(SM.SourceCat.kill==1)
            PredictData=PM.predictKernelPolCluster(LM.VS.ThisDataChunk,LM.SM,ApplyTimeJones=Jones)
            SM.RestoreCat()


            # import pylab
            # pylab.clf()
            # nbl=1#LM.VS.MS.nbl
            # a=LM.VS.ThisDataChunk["data"][1::nbl,:,:].flatten().real
            # b=PredictData[1::nbl,:,:].flatten().real
            # pylab.plot(a)
            # pylab.plot(b)
            # pylab.plot(a-b)
            # pylab.draw()
            # pylab.show(False)
            # stop

            LM.VS.ThisDataChunk["data"]-=PredictData

            if ApplyCal!=None:
                PM.ApplyCal(LM.VS.ThisDataChunk,Jones,ApplyCal)


            LM.VS.MS.data=LM.VS.ThisDataChunk["data"]

            

            LM.VS.MS.SaveVis(Col=WriteColName)

        if Res=="EndChunk":
            Load=VS.LoadNextVisChunk()
            if Load=="EndOfObservation":
                break


    NpShared.DelAll()

    



def Restore(options=None):
    import ClassMS
    if options==None:
        f = open("last_killMS.obj",'rb')
        options = pickle.load(f)
    MS=ClassMS.ClassMS(options.ms)
    MS.Restore()

#     KalmanKill.set_chanOptions(options)
#     #KalmanKill.zero_chans(options)

if __name__=="__main__":
    read_options()
    f = open("last_killMS.obj",'rb')
    options = pickle.load(f)
    if options.DoBar=="0":
        from progressbar import ProgressBar
        ProgressBar.silent=1
    # else:
    #     os.system('clear')

    if options.Restore=="1":
        Restore(options)
    else:
        logo.print_logo()
        main(options=options)
