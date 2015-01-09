#!/usr/bin/env python

import optparse
import sys
import MyPickle
import logo

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]

# test

#import numpy
#print numpy.__file__
#import pyrap
#print pyrap.__file__
#stop


import ModColor
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
import ClassSM
from ClassLM import ClassLM
import ClassTimeIt
import ClassVisServer
from PredictGaussPoints_NumExpr import ClassPredict
import ModLinAlg

import MyLogger
log=MyLogger.getLogger("killMS")

def read_options():
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--ms',help='Input MS to draw [no default]',default='')
    group.add_option('--SkyModel',help='List of targets [no default]',default='')
    opt.add_option_group(group)
    
    # group = optparse.OptionGroup(opt, "* Data selection options", "ColName is set to DATA column by default, and other parameters select all the data.")
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--kills',help='Name or number index of sources to kill',default="")
    group.add_option('--invert',help='Invert the selected sources to kill',default="0")
    opt.add_option_group(group)
    
    group = optparse.OptionGroup(opt, "* Algorithm options", "Default values should give reasonable results, but all of them have noticeable influence on the results")
    group.add_option('--timestep',help='Time interval for a solution [minutes]. Default is %default. ',default=30)
    group.add_option('--NCPU',help=' Number of cores to use for the calibration of the Tikhonov output. Default is %default ',default="6")
    group.add_option('--niter',help=' Number of iterations for the solve. Default is %default ',default="20")
    #group.add_option('--doSmearing',help='Takes time and frequency smearing if enabled. Default is %default ',default="0")
    group.add_option('--PolMode',help=' Polarisation mode (Scalar/HalfFull). Default is %default',default="Scalar")
    #group.add_option('--ChanSels',help=' Channel selection. Default is %default',default="")
    #group.add_option('--BLFlags',help=' Baselines To be flagged. Default is %default',default="")
    group.add_option('--Restore',help=' Restore BACKUP in CORRECTED. Default is %default',default="0")
    #group.add_option('--LOFARBeamParms',help='Applying the LOFAR beam parameters [Mode[A,AE,E],TimeStep]. Default is %default',default="")
    group.add_option('--LOFARBeamParms',help='Not Working yet',default="")

    group.add_option('--TChunk',help=' Time Chunk in hours. Default is %default',default="15")
    group.add_option('--SubOnly',help=' Only substract the skymodel. Default is %default',default="0")
    group.add_option('--DoBar',help=' Draw progressbar. Default is %default',default="1")
    group.add_option('--InCol',help=' Column to work on. Default is %default',default="CORRECTED_DATA_BACKUP")
    group.add_option('--ApplyCal',help=' Apply direction averaged gains to residual data. Default is %default',default="0")
    


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

    TChunk=float(options.TChunk)
    delta_time=float(options.timestep)
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

        
    ReadColName="DATA"#options.InCol
    WriteColName="CORRECTED_DATA"

    SM=ClassSM.ClassSM(options.SkyModel,
                       killdirs=kills,invert=invert)
    
    VS=ClassVisServer.ClassVisServer(options.ms,ColName=ReadColName,
                                     TVisSizeMin=delta_time,TChunkSize=TChunk)
    

    # LM=ClassLM(VS,SM,PolMode="HalfFull")
    LM=ClassLM(VS,SM,PolMode=options.PolMode,
               NIter=niterin,NCPU=NCPU)

    PM=ClassPredict()
    SM=LM.SM

    while True:
        Res=LM.setNextData()

        if Res==True:
            # if not(SubOnly):
            #     LM.doNextTimeSolve_Parallel()
            LM.doNextTimeSolve_Parallel()
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

            LM.VS.ThisDataChunk["data"]-=PredictData
            LM.VS.MS.data=LM.VS.ThisDataChunk["data"]
            LM.VS.MS.SaveVis(Col=WriteColName)

        if Res=="EndChunk":
            Load=VS.LoadNextVisChunk()
            if Load=="EndOfObservation":
                break




    



def Restore(options=None):
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
