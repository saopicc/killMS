#!/usr/bin/env python
#turtles
import optparse
import sys
from killMS2.Other import MyPickle
from killMS2.Other import logo
from killMS2.Other import ModColor
from killMS2.Other import MyLogger
from killMS2.Other import MyPickle
from killMS2.Other import PrintOptParse
from killMS2.Parset import MyOptParse
import numpy as np


# log
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
    


#from killMS2.Data import MergeJones
from killMS2.Data import ClassJonesDomains
import time
import os
import numpy as np
import pickle
from SkyModel.Sky import ClassSM
from killMS2.Wirtinger.ClassWirtingerSolver import ClassWirtingerSolver

from killMS2.Other import ClassTimeIt
from killMS2.Data import ClassVisServer

from Predict.PredictGaussPoints_NumExpr5 import ClassPredictParallel as ClassPredict 
#from Predict.PredictGaussPoints_NumExpr5 import ClassPredict as ClassPredict 
#from Predict.PredictGaussPoints_NumExpr2 import ClassPredictParallel as ClassPredict_orig
#from Predict.PredictGaussPoints_NumExpr4 import ClassPredict as ClassPredict 
#from Predict.PredictGaussPoints_NumExpr2 import ClassPredict as ClassPredict_orig
#from Sky.PredictGaussPoints_NumExpr4 import ClassPredict as ClassPredict 

#from Sky.PredictGaussPoints_NumExpr2 import ClassPredictParallel as ClassPredict_orig 
#from Sky.PredictGaussPoints_NumExpr3 import ClassPredict as ClassPredict 
#from Sky.PredictGaussPoints_NumExpr2 import ClassPredict as ClassPredict_orig 

from killMS2.Array import ModLinAlg
from killMS2.Array import NpShared
from killMS2.Other import reformat

import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

from killMS2.Parset import ReadCFG

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
    #OP.add_option('PredictColName',type="str",help=' . Default is %default')
    OP.add_option('FreePredictColName',type="str",help=' . Default is %default')
    OP.add_option('FreePredictGainColName',type="str",help=' . Default is %default')



    OP.OptionGroup("* Sky catalog related options","SkyModel")
    OP.add_option('SkyModel',help='List of targets [no default]')
    #OP.add_option('LOFARBeam',help='(Mode, Time): Mode can be AE, E, or A for "Array factor" and "Element beam". Time is the estimation time step')
    OP.add_option('kills',help='Name or number index of sources to kill')
    OP.add_option('invert',help='Invert the selected sources to kill')
    OP.add_option('Decorrelation',type="str",help=' . Default is %default')
    OP.add_option('FreeFullSub',type="int",help=' . Default is %default')


    OP.OptionGroup("* Sky image related options","ImageSkyModel")
    OP.add_option('BaseImageName')
    OP.add_option('ImagePredictParset')
    OP.add_option('OverS')
    OP.add_option('wmax')
    OP.add_option('MaskImage')
    OP.add_option('NodesFile')

    OP.OptionGroup("* Data Selection","DataSelection")
    OP.add_option('UVMinMax',help='Baseline length selection in km. For example UVMinMax=0.1,100 selects baseline with length between 100 m and 100 km. Default is %default')
    OP.add_option('FlagAnts',type="str",help='FlagAntenna patern. Default is %default')
    OP.add_option('DistMaxToCore',type="float",help='Maximum distance to core in km. Default is %default')
    OP.add_option('FillFactor',type="float")
    OP.add_option('FieldID',type="int")
    OP.add_option('DDID',type="int")

    OP.OptionGroup("* Beam Options","Beam")
    OP.add_option('BeamModel',type="str",help='Apply beam model, Can be set to: None/LOFAR. Default is %default')
    OP.add_option('LOFARBeamMode',type="str",help='LOFAR beam mode. "AE" sets the beam model to Array and Element. Default is %default')
    OP.add_option('DtBeamMin',type="float",help='Estimate the beam every this interval [in minutes]. Default is %default')
    OP.add_option('CenterNorm',type="str",help='Normalise the beam at the field center. Default is %default')
    OP.add_option('NChanBeamPerMS',type="int",help='Number of channel in the Beam Jones matrix. Default is %default')

    OP.OptionGroup("* PreApply killMS Solutions","PreApply")
    OP.add_option('PreApplySols',type="str",help='Pre-apply killMS solutions in the predict step. Has to be a list. Default is %default')
    OP.add_option('PreApplyMode',type="str",help='Mode for the pre-applied killMS solutions ("A", "P" and "AP" for Amplitude, Phase and Amplitude+Phase). Has to be a list. Default is %default')


    OP.OptionGroup("* Weighting scheme","Weighting")
    OP.add_option('Resolution',type="float",help='Resolution in arcsec. Default is %default')
    OP.add_option('Weighting',type="str",help='Weighting scheme. Default is %default')
    OP.add_option('Robust',type="float",help='Briggs Robust parameter. Default is %default')
    
    OP.OptionGroup("* Action options","Actions")
    OP.add_option('DoPlot',type="int",help='Plot the solutions, for debugging. Default is %default')
    OP.add_option('SubOnly',type="int",help='Substact selected sources. Default is %default')
    OP.add_option('DoBar',help=' Draw progressbar. Default is %default',default="1")
    OP.add_option('NCPU',type="int",help='Number of cores to use. Default is %default ')

    # OP.OptionGroup("* PreApply Solution-related options","PreApply")
    # OP.add_option('PreApplySols')#,help='Solutions to apply to the data before solving.')
    # #OP.add_option('PreApplyMode',help='Solutions to apply to the data before solving.')

    OP.OptionGroup("* Solution-related options","Solutions")
    OP.add_option('ExtSols',type="str",help='External solution file. If set, will not solve.')
    #OP.add_option('ApplyMode',type="str",help='Substact selected sources. ')
    OP.add_option('ClipMethod',type="str",help='Clip data in the IMAGING_WEIGHT column. Can be set to Resid or DDEResid . Default is %default')
    OP.add_option('OutSolsName',type="str",help='If specified will save the estimated solutions in this file. Default is %default')
    OP.add_option('ApplyCal',type="int",help='Apply direction averaged gains to residual data in the mentioned direction. \
    If ApplyCal=-1 takes the mean gain over directions. -2 if off. Default is %default')
    
    OP.OptionGroup("* Solver options","Solvers")
    OP.add_option('SolverType',help='Name of the solver to use (CohJones/KAFCA)')
    OP.add_option('PrecisionDot',help='Dot product Precision (S/D). Default is %default.',type="str")
    OP.add_option('PolMode',help='Polarisation mode (Scalar/IFull). Default is %default')
    OP.add_option('dt',type="float",help='Time interval for a solution [minutes]. Default is %default. ')
    OP.add_option('NChanSols',type="int",help='Number of solutions along frequency axis. Default is %default. ')
    


    OP.OptionGroup("* CohJones additional options","CohJones")
    OP.add_option('NIterLM',type="int",help=' Number of iterations for the solve. Default is %default ')
    OP.add_option('LambdaLM',type="float",help=' Lambda parameter for CohJones. Default is %default ')
    OP.add_option('LambdaTk',type="float",help=' Tikhonov regularisation parameter. Default is %default')
    

    OP.OptionGroup("* KAFCA additional options","KAFCA")
    OP.add_option('NIterKF',type="int",help=' Number of iterations for the solve. Default is %default ')
    OP.add_option('LambdaKF',type="float",help=' Lambda parameter for KAFCA. Default is %default ')
    OP.add_option('InitLM',type="int",help='Initialise Kalman filter with Levenberg Maquardt. Default is %default')
    OP.add_option('InitLMdt',type="float",help='Time interval in minutes. Default is %default')
    OP.add_option('CovP',type="float",help='Initial prior Covariance in fraction of the initial gain amplitude. Default is %default') 
    OP.add_option('CovQ',type="float",help='Intrinsic process Covariance in fraction of the initial gain amplitude. Default is %default') 
    OP.add_option('evPStep',type="int",help='Start calculation evP every evP_Step after that step. Default is %default')
    OP.add_option('evPStepStart',type="int",help='Calcule (I-KJ) matrix every evP_Step steps. Default is %default')
    

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
    DoApplyCal=(options.ApplyCal!=-2)
    if type(options.ClipMethod)!=list: stop

    ReWeight=(len(options.ClipMethod)>0)


    if MSName!=None:
        options.MSName=MSName

    if options.MSName=="":
        print "Give an MS name!"
        exit()

    # if options.SkyModel=="":
    #     print "Give a Sky Model!"
    #     exit()
    # if not(".npy" in options.SkyModel):
    #     print "Give a numpy sky model!"
    #     exit()

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

    SolsName=options.SolverType
    if options.OutSolsName!="":
        #FileName="%s%s"%(reformat.reformat(options.MSName),options.OutSolsName)
        #if not(FileName[-4::]==".npz"): FileName+=".npz"
        SolsName=options.OutSolsName

    ParsetName="%skillMS.%s.sols.parset"%(reformat.reformat(options.MSName),SolsName)
    OP.ToParset(ParsetName)

    GD=OP.DicoConfig
    if GD["ImageSkyModel"]["BaseImageName"]=="":
        print>>log,"Predict Mode: Catalog"
        PredictMode="Catalog"
    else:
        PredictMode="Image"
        BaseImageName=GD["ImageSkyModel"]["BaseImageName"]
        ParsetName=GD["ImageSkyModel"]["ImagePredictParset"]
        if ParsetName=="":
            ParsetName="%s.parset"%BaseImageName
        print>>log,"Predict Mode: Image, with Parset: %s"%ParsetName
        GDPredict=ReadCFG.Parset(ParsetName).DicoPars

        if not("PSFFacets" in GDPredict["ImagerGlobal"].keys()):
               GDPredict["ImagerGlobal"]["PSFFacets"]=0
               GDPredict["ImagerGlobal"]["PSFOversize"]=1


        GDPredict["Compression"]["CompDeGridMode"]=False
        #GDPredict["Compression"]["CompDeGridMode"]=True
        
        if options.OverS!=None:
            GDPredict["ImagerCF"]["OverS"]=options.OverS
        if options.wmax!=None:
            GDPredict["ImagerCF"]["wmax"]=options.wmax
        GD["GDImage"]=GDPredict
        GDPredict["GDkMS"]=GD


    #SM.SourceCat.I*=1000**2
    VS=ClassVisServer.ClassVisServer(options.MSName,ColName=ReadColName,
                                     TVisSizeMin=dt,
                                     DicoSelectOptions=DicoSelectOptions,
                                     TChunkSize=TChunk,IdSharedMem=IdSharedMem,
                                     NCPU=NCPU,
                                     Weighting=options.Weighting,
                                     Robust=options.Robust,
                                     GD=GD)

    print VS.MS
    if not(WriteColName in VS.MS.ColNames):
        print>>log, "Column %s not in MS "%WriteColName
        exit()
    if not(ReadColName in VS.MS.ColNames):
        print>>log, "Column %s not in MS "%ReadColName
        exit()

    if PredictMode=="Catalog":
        SM=ClassSM.ClassSM(options.SkyModel,
                           killdirs=kills,
                           invert=invert)
        SM.Type="Catalog"

    else:
        from killMS2.Predict import ClassImageSM2 as ClassImageSM
        #from killMS2.Predict import ClassImageSM3 as ClassImageSM
        PreparePredict=ClassImageSM.ClassPreparePredict(BaseImageName,VS,GD=GDPredict,DoDeconvolve=False,IdSharedMem=IdSharedMem)
        SM=PreparePredict.SM
        #VS.setGridProps(PreparePredict.FacetMachine.Cell,PreparePredict.FacetMachine.NpixPaddedFacet)
        VS.setGridProps(PreparePredict.FacetMachine.Cell,None)#PreparePredict.FacetMachine.NpixPaddedFacet)
        FacetMachine=PreparePredict.FacetMachine
        VS.setFOV(FacetMachine.OutImShape,FacetMachine.PaddedGridShape,FacetMachine.FacetShape,FacetMachine.CellSizeRad)
    VS.setSM(SM)
    VS.CalcWeigths()
    
        


    # BeamProps=None
    # if options.LOFARBeam!="":
    #     Mode,sTimeMin=options.LOFARBeam.split(",")
    #     TimeMin=float(sTimeMin)
    #     BeamProps=Mode,TimeMin

    ResolutionRad=(options.Resolution/3600)*(np.pi/180)
    ConfigJacobianAntenna={"DoSmearing":DoSmearing,
                           "ResolutionRad":ResolutionRad,
                           "LambdaKF":options.LambdaKF,
                           "LambdaLM":options.LambdaLM,
                           "DoReg":False,#True,
                           "gamma":1,
                           "AmpQx":.5,
                           "PrecisionDot":options.PrecisionDot}

    if (options.SolverType=="KAFCA"):
        NIter=options.NIterKF
    elif options.SolverType=="CohJones":
        NIter=options.NIterLM

    Solver=ClassWirtingerSolver(VS,SM,PolMode=options.PolMode,
                                #BeamProps=BeamProps,
                                NIter=NIter,
                                NCPU=NCPU,
                                SolverType=options.SolverType,
                                evP_Step=options.evPStep,evP_StepStart=options.evPStepStart,
                                DoPlot=options.DoPlot,
                                DoPBar=options.DoBar,
                                IdSharedMem=IdSharedMem,
                                ConfigJacobianAntenna=ConfigJacobianAntenna,
                                GD=GD)
    
    
    Solver.InitSol(TestMode=False)

    PM=ClassPredict(NCPU=NCPU,IdMemShared=IdSharedMem,DoSmearing=DoSmearing)
    PM2=None#ClassPredict_orig(NCPU=NCPU,IdMemShared=IdSharedMem)


    Solver.InitMeanBeam()
    if (options.SolverType=="KAFCA"):

        if (options.InitLM):
            rms,SolverInit_G=GiveNoise(options,
                                       DicoSelectOptions,
                                       IdSharedMem,
                                       SM,PM,PM2,ConfigJacobianAntenna,GD)
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

    if SM.Type=="Catalog":
        ind=np.where(SM.SourceCat.kill==1)[0]
        DoSubstract=(ind.size>0)
    else:
        DoSubstract=0
    


    # print "!!!!!!!!!!!!!!"
    #
    # Solver.InitCovariance(FromG=True,sigP=options.CovP,sigQ=options.CovQ)

    SourceCatSub=None
    SaveSols=False
    # # ##############################
    # # Catch numpy warning
    # np.seterr(all='raise')
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    # # ##############################

    while True:

        Load=VS.LoadNextVisChunk()
        if Load=="EndOfObservation":
            break

        if options.ExtSols=="":
            SaveSols=True
            Solver.doNextTimeSolve_Parallel()
            #Solver.doNextTimeSolve_Parallel(SkipMode=True)
            #Solver.doNextTimeSolve()#SkipMode=True)
            
            def SavePredict(ArrayName,FullPredictColName):
                print>>log, "Writing full predicted data in column %s of %s"%(FullPredictColName,options.MSName)
                VS.MS.AddCol(FullPredictColName)
                PredictData=NpShared.GiveArray("%s%s"%(IdSharedMem,ArrayName))

                t=VS.MS.GiveMainTable(readonly=False)#table(VS.MS.MSName,readonly=False,ack=False)
                t.putcol(FullPredictColName,VS.MS.ToOrigFreqOrder(PredictData),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.close()


            FreePredictGainColName=GD["VisData"]["FreePredictGainColName"]
            if (FreePredictGainColName!=None):
                ArrayName="PredictedDataGains"
                FullPredictColName=FreePredictGainColName
                SavePredict(ArrayName,FullPredictColName)

            FreePredictColName=GD["VisData"]["FreePredictColName"]
            if (FreePredictColName!=None):
                ArrayName="PredictedData"
                FullPredictColName=FreePredictColName
                SavePredict(ArrayName,FullPredictColName)

            if GD["SkyModel"]["FreeFullSub"]:
                print>>log, "Substracting free predict from data"
                PredictData=NpShared.GiveArray("%s%s"%(IdSharedMem,"PredictedDataGains"))
                Solver.VS.ThisDataChunk["data"]-=PredictData
                print>>log, "  save visibilities in %s column"%WriteColName
                t=Solver.VS.MS.GiveMainTable(readonly=False)#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                t.putcol(WriteColName,VS.MS.ToOrigFreqOrder(Solver.VS.MS.data),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.close()


            Sols=Solver.GiveSols(SaveStats=True)

            # ##########
            # FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),SolsName)

            # print>>log, "Save Solutions in file: %s"%FileName
            # Sols=Solver.GiveSols()
            # StationNames=np.array(Solver.VS.MS.StationNames)
            # np.savez(FileName,
            #          Sols=Sols,
            #          StationNames=StationNames,
            #          SkyModel=SM.ClusterCat,
            #          ClusterCat=SM.ClusterCat,
            #          SourceCatSub=SourceCatSub,
            #          ModelName=options.SkyModel)

            SolsFreqDomain=VS.SolsFreqDomains
            if SaveSols:

                FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),SolsName)

                print>>log, "Save Solutions in file: %s"%FileName
                Sols=Solver.GiveSols()
                SolsSave=Sols
                ClusterCat=SM.ClusterCat
                VS.BeamTimes
                if SM.Type=="Image":
                    nt,nch,na,nd,_,_=Sols.G.shape
                    nd=PreparePredict.NDirsOrig
                    SolsAll=np.zeros((nt,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(nch,na,nd,2,2)),("Stats",np.float32,(nch,na,4))])
                    SolsAll=SolsAll.view(np.recarray)
                    SolsAll.G[:,:,:,:,0,0]=1
                    SolsAll.G[:,:,:,:,1,1]=1
                    SolsAll.G[:,:,:,PreparePredict.MapClusterCatOrigToCut,:,:]=Sols.G[:,:,:,:,:,:]
                    SolsAll.t0=Sols.t0
                    SolsAll.t1=Sols.t1
                    SolsAll.Stats=Sols.Stats
                    SolsSave=SolsAll
                    ClusterCat=PreparePredict.ClusterCatOrig

                StationNames=np.array(Solver.VS.MS.StationNames)
                
                np.savez(FileName,
                         Sols=SolsSave,
                         StationNames=StationNames,
                         SkyModel=ClusterCat,
                         ClusterCat=ClusterCat,
                         SourceCatSub=SourceCatSub,
                         ModelName=options.SkyModel,
                         FreqDomains=VS.SolsFreqDomains,
                         BeamTimes=VS.BeamTimes)

        else:
            DicoLoad=np.load(options.ExtSols)
            Sols=DicoLoad["Sols"]
            Sols=Sols.view(np.recarray)
            SolsFreqDomain=DicoLoad["FreqDomains"]
            
        # substract
        #ind=np.where(SM.SourceCat.kill==1)[0]
        if ((DoSubstract)|(DoApplyCal)|(ReWeight)):
            Jones={}
            Jones["t0"]=Sols.t0
            Jones["t1"]=Sols.t1
            Jones["FreqDomain"]=SolsFreqDomain
            nt,nch,na,nd,_,_=Sols.G.shape
            G=np.swapaxes(Sols.G,1,3).reshape((nt,nd,na,nch,2,2))
            G=np.require(G, dtype=np.complex64, requirements="C_CONTIGUOUS")

            # if not("A" in options.ApplyMode):
            #     gabs=np.abs(G)
            #     gabs[gabs==0]=1.
            #     G/=gabs


            Jones["Jones"]=G
            Jones["JonesH"]=ModLinAlg.BatchH(Jones["Jones"])

            try:
                Jones["Stats"]=Sols.Stats
            except:
                Jones["Stats"]=None

            # Jones["ChanMap"]=VS.VisToJonesChanMapping
            times=Solver.VS.ThisDataChunk["times"]
            freqs=Solver.VS.ThisDataChunk["freqs"]
            DomainMachine=ClassJonesDomains.ClassJonesDomains()

            if options.BeamModel==None:
                JonesMerged=Jones
            else:
                Jones["tm"]=(Jones["t0"]+Jones["t1"])/2.
                PreApplyJones=Solver.VS.ThisDataChunk["PreApplyJones"]
                PreApplyJones["tm"]=(PreApplyJones["t0"]+PreApplyJones["t1"])/2.
                DomainsMachine=ClassJonesDomains.ClassJonesDomains()
                JonesMerged=DomainsMachine.MergeJones(Jones,PreApplyJones)
                
                DicoJonesMatrices=JonesMerged


            DomainMachine.AddVisToJonesMapping(JonesMerged,times,freqs)
            JonesMerged["JonesH"]=ModLinAlg.BatchH(JonesMerged["Jones"])


            if ("Resid" in options.ClipMethod)|("DDEResid" in options.ClipMethod):
                print>>log, ModColor.Str("Clipping bad solution-based data ... ",col="green")
                

                nrows=Solver.VS.ThisDataChunk["times"].size

                Solver.VS.ThisDataChunk["W"]=np.ones((nrows,Solver.VS.MS.ChanFreq.size),np.float64)


                ################
                # #PM.GiveCovariance(Solver.VS.ThisDataChunk,Jones)

                print>>log,"   Compute residual data"
                Predict=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=JonesMerged)
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
                    PM.GiveCovariance(Solver.VS.ThisDataChunk,JonesMerged,SM)



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
                VS.MS.AddCol("IMAGING_WEIGHT")
                t=Solver.VS.MS.GiveMainTable(readonly=False)#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                t.putcol("IMAGING_WEIGHT",VS.MS.ToOrigFreqOrder(Weights),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.close()



            if "ResidAnt" in options.ClipMethod:
                print>>log,"Compute weighting based on antenna-selected residual"
                DomainMachine.AddVisToJonesMapping(Jones,times,freqs)
                nrows=Solver.VS.ThisDataChunk["times"].size
                Solver.VS.ThisDataChunk["W"]=np.ones((nrows,Solver.VS.MS.ChanFreq.size),np.float64)
                PM.GiveCovariance(Solver.VS.ThisDataChunk,Jones,SM,Mode="ResidAntCovariance")

                Weights=Solver.VS.ThisDataChunk["W"]
                Weights/=np.mean(Weights)
                print>>log, "  Writting in IMAGING_WEIGHT column "
                t=Solver.VS.MS.GiveMainTable(readonly=False)#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                t.putcol("IMAGING_WEIGHT",VS.MS.ToOrigFreqOrder(Weights),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.close()

            if DoSubstract:
                print>>log, ModColor.Str("Substract sources ... ",col="green")
                SM.SelectSubCat(SM.SourceCat.kill==1)

                SourceCatSub=SM.SourceCat.copy()

                if options.SubOnly==1:
                    print>>log, ModColor.Str(" Sublonly ... ",col="green")
                    PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM)
                else:
                    PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=JonesMerged)
                    
                    PredictColName=options.PredictColName
                    if PredictColName!="":
                        print>>log, "Writing predicted data in column %s of %s"%(PredictColName,MSName)
                        VS.MS.AddCol(PredictColName)
                        t=Solver.VS.MS.GiveMainTable(readonly=False)#table(VS.MS.MSName,readonly=False,ack=False)
                        t.putcol(PredictColName,VS.MS.ToOrigFreqOrder(PredictData),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                        t.close()
                    
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

            if DoApplyCal:
                print>>log, ModColor.Str("Apply calibration in direction: %i"%options.ApplyCal,col="green")
                PM.ApplyCal(Solver.VS.ThisDataChunk,JonesMerged,options.ApplyCal)

            Solver.VS.MS.data=Solver.VS.ThisDataChunk["data"]
            Solver.VS.MS.flags_all=Solver.VS.ThisDataChunk["flags"]
            # Solver.VS.MS.SaveVis(Col=WriteColName)

            if (DoSubstract|DoApplyCal):
                print>>log, "Save visibilities in %s column"%WriteColName
                t=Solver.VS.MS.GiveMainTable(readonly=False)#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                t.putcol(WriteColName,VS.MS.ToOrigFreqOrder(Solver.VS.MS.data),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.putcol("FLAG",VS.MS.ToOrigFreqOrder(Solver.VS.MS.flags_all),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                t.close()

                



    NpShared.DelAll(IdSharedMem)

    
def GiveNoise(options,DicoSelectOptions,IdSharedMem,SM,PM,PM2,ConfigJacobianAntenna,GD):
    print>>log, ModColor.Str("Initialising Kalman filter with Levenberg-Maquardt estimate")
    dtInit=float(options.InitLMdt)
    VSInit=ClassVisServer.ClassVisServer(options.MSName,ColName=options.InCol,
                                         TVisSizeMin=dtInit,
                                         DicoSelectOptions=DicoSelectOptions,
                                         TChunkSize=dtInit/60,IdSharedMem=IdSharedMem,
                                         SM=SM,NCPU=options.NCPU,GD=GD)
    
    VSInit.setSM(SM)
    VSInit.CalcWeigths()
    VSInit.LoadNextVisChunk()
    # # test
    # PredictData=PM.predictKernelPolCluster(VSInit.ThisDataChunk,SM)
    # PredictData2=PM2.predictKernelPolCluster(VSInit.ThisDataChunk,SM)
    # print np.max(PredictData-PredictData2)
    # stop
    # #######
    SolverInit=ClassWirtingerSolver(VSInit,SM,PolMode=options.PolMode,
                                    NIter=options.NIterLM,NCPU=options.NCPU,
                                    SolverType="CohJones",
                                    #DoPlot=options.DoPlot,
                                    DoPBar=False,IdSharedMem=IdSharedMem,
                                    ConfigJacobianAntenna=ConfigJacobianAntenna,GD=GD)
    SolverInit.InitSol(TestMode=False)
    SolverInit.doNextTimeSolve_Parallel(OnlyOne=True)
    #SolverInit.doNextTimeSolve()
    Sols=SolverInit.GiveSols()
    Jones={}
    Jones["t0"]=Sols.t0
    Jones["t1"]=Sols.t1
    nt,na,nd,_,_=Sols.G.shape
    G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
    Jones["Jones"]=G
    Jones["JonesH"]=ModLinAlg.BatchH(G)
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
    #os.system('clear')
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
    MSName=options.MSName

    if type(MSName)==list:
        lMS=MSName
        print>>log, "In batch mode, running killMS on the following MS:"
        for MS in lMS:
            print>>log, "  %s"%MS
        
    elif ".txt" in MSName:
        f=open(MSName)
        Ls=f.readlines()
        f.close()
        MSName=[]
        for l in Ls:
            ll=l.replace("\n","")
            MSName.append(ll)
        lMS=MSName
        print>>log, "In batch mode, running killMS on the following MS:"
        for MS in lMS:
            print>>log, "  %s"%MS
    elif "*" in options.MSName:
        Patern=options.MSName
        lMS=sorted(glob.glob(Patern))
        print>>log, "In batch mode, running killMS on the following MS:"
        for MS in lMS:
            print>>log, "  %s"%MS
    else:
        lMS=[options.MSName]

    
 
    try:

        #print MSName
        for MSName in lMS:
            main(OP=OP,MSName=MSName)
    except:
        NpShared.DelAll(IdSharedMem)
            
