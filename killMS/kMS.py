#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
#!/usr/bin/env python
#turtles
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import time
import subprocess
import killMS
if "PYTHONPATH_FIRST" in os.environ.keys() and int(os.environ["PYTHONPATH_FIRST"]):
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


# hack to allow 'from killMS... import...'
#sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

# # ##############################
# # Catch numpy warning
# import numpy as np
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################

# log
from DDFacet.Other import logger, ModColor

log = logger.getLogger("killMS")
#log.setLevel(logger.logging.CRITICAL)


SaveFile="last_killMS.obj"


if "nocol" in sys.argv:
    print("nocol")
    ModColor.silent=1
if "nox" in sys.argv:
    import matplotlib
    matplotlib.use('agg')
    print(ModColor.Str(" == !NOX! =="))
    

IdSharedMem = None

import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

from killMS.Parset import ReadCFG, MyOptParse

parset_path = os.path.join(os.path.dirname(killMS.__file__), "Parset", "DefaultParset.cfg")
    #
    # os.path.join(os.environ["KILLMS_DIR"], "killMS", "killMS", "Parset", "DefaultParset.cfg")
print(parset_path)
if not os.path.exists(parset_path):
    raise IOError("Default parset could not be located in {0:s}. Check your installation".format(parset_path))
Parset = ReadCFG.Parset(parset_path)


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
    OP.add_option('Parallel',type="int",help=' . Default is %default')


    OP.OptionGroup("* Sky catalog related options","SkyModel")
    OP.add_option('SkyModel',help='List of targets [no default]')
    #OP.add_option('LOFARBeam',help='(Mode, Time): Mode can be AE, E, or A for "Array factor" and "Element beam". Time is the estimation time step')
    OP.add_option('kills',help='Name or number index of sources to kill')
    OP.add_option('invert',help='Invert the selected sources to kill')
    OP.add_option('Decorrelation',type="str",help=' . Default is %default')
    OP.add_option('FreeFullSub',type="int",help=' . Default is %default')
    OP.add_option('SkyModelCol',type="str",help=' . Default is %default')


    OP.OptionGroup("* Sky image related options","ImageSkyModel")
    OP.add_option('BaseImageName')
    OP.add_option('ImagePredictParset')
    OP.add_option('DicoModel')
    OP.add_option('OverS')
    OP.add_option('wmax')
    OP.add_option('MaskImage')
    OP.add_option('NodesFile')
    OP.add_option('MaxFacetSize')
    OP.add_option('MinFacetSize')
    OP.add_option('DDFCacheDir')
    OP.add_option('RemoveDDFCache')
    OP.add_option('FilterNegComp')
    OP.add_option('ThSolve',help="If the tessel has an apparant SumFlux bellow ThSolve*MaxSumFlux (Max over tessels), it will be unsolved (J=1)")

    OP.OptionGroup("* Compression","Compression")
    OP.add_option('CompressionMode',help='Only Auto implemented. Default is %default')
    OP.add_option('CompressionDirFile',help='Directions in which to do the compression. Default is %default')
    OP.add_option('MergeStations',help='Merge stations into a single one. Use --MergeStations=[CS] to merge all core stations. Default is %default')
    
    OP.OptionGroup("* Data Selection","DataSelection")
    OP.add_option('UVMinMax',help='Baseline length selection in km. For example UVMinMax=0.1,100 selects baseline with length between 100 m and 100 km. Default is %default')
    OP.add_option('ChanSlice',type="str",help='Channel selection option. Default is %default')
    OP.add_option('FlagAnts',type="str",help='FlagAntenna patern. Default is %default')
    OP.add_option('DistMaxToCore',type="float",help='Maximum distance to core in km. Default is %default')
    OP.add_option('FillFactor',type="float")
    OP.add_option('FieldID',type="int")
    OP.add_option('DDID',type="int")

    OP.OptionGroup("* Beam Options","Beam")
    OP.add_option('BeamModel',type="str",help='Apply beam model, Can be set to: None/LOFAR. Default is %default')
    OP.add_option('BeamAt',type="str",help='Where to apply beam model, Can be set to: tessel/facet. Default is %default')
    OP.add_option('LOFARBeamMode',type="str",help='LOFAR beam mode. "AE" sets the beam model to Array and Element. Default is %default')
    OP.add_option('DtBeamMin',type="float",help='Estimate the beam every this interval [in minutes]. Default is %default')
    OP.add_option('CenterNorm',type="str",help='Normalise the beam at the field center. Default is %default')
    OP.add_option('NChanBeamPerMS',type="int",help='Number of channel in the Beam Jones matrix. Default is %default')
    OP.add_option('FITSParAngleIncDeg',type="float",help='Estimate the beam every this PA change [in deg]. Default is %default')
    OP.add_option('FITSFile',type="str",help='FITS beam mode filename template. Default is %default')
    OP.add_option('FITSLAxis',type="str",help='L axis of FITS beam. Default is %default')
    OP.add_option('FITSMAxis',type="str",help='L axis of FITS beam. Default is %default')
    OP.add_option('FITSFeed',type="str",help='FITS feed. xy or rl or None to take from MS. Default is %default')
    OP.add_option('FITSFeedSwap',type="int",help='Swap the feeds around. Default is %default')
    OP.add_option('FITSVerbosity',type="int",help='Verbosity of debug messages. Default is %default')
    OP.add_option("ApplyPJones",type="int",help='derotate visibility data (only when FITS beam is active and also time sampled)')
    OP.add_option("FlipVisibilityHands",type="int",help='apply anti-diagonal matrix if FITS beam is enabled effectively swapping X and Y or R and L and their respective hands')
    OP.add_option('FeedAngle',type="float",help='offset feed angle to add to parallactic angle')
    OP.add_option('FITSFrame', type='str', help=' coordinate frame for FITS beams. Currently, alt-az, equatorial and zenith mounts are supported. #options:altaz|altazgeo|equatorial|zenith . Default is %default')

    OP.OptionGroup("* PreApply killMS Solutions","PreApply")
    OP.add_option('PreApplySols',type="str",help='Pre-apply killMS solutions in the predict step. Has to be a list. Default is %default')
    OP.add_option('PreApplyMode',type="str",help='Mode for the pre-applied killMS solutions ("A", "P" and "AP" for Amplitude, Phase and Amplitude+Phase). Has to be a list. Default is %default')


    OP.OptionGroup("* Weighting scheme","Weighting")
    OP.add_option('Resolution',type="float",help='Resolution in arcsec. Default is %default')
    OP.add_option('WeightInCol',type="str",help='Weighting column to take into account to weight the visibilities in the solver. Default is %default')
    OP.add_option('Weighting',type="str",help='Weighting scheme. Default is %default')
    OP.add_option('Robust',type="float",help='Briggs Robust parameter. Default is %default')
    OP.add_option('WeightUVMinMax',help='Baseline length selection in km for full weight. For example WeightUVMinMax=0.1,100 selects baseline with length between 100 m and 100 km. Default is %default')
    OP.add_option('WTUV',type="float",help='Scaling factor to apply to weights outside range of WeightUVMinMax. Default is %default')
    
    OP.OptionGroup("* Action options","Actions")
    OP.add_option('DoPlot',type="int",help='Plot the solutions, for debugging. Default is %default')
    OP.add_option('SubOnly',type="int",help='Subtract selected sources. Default is %default')
    OP.add_option('DoBar',help=' Draw progressbar. Default is %default',default="1")
    OP.add_option('NCPU',type="int",help='Number of cores to use. Default is %default ')
    OP.add_option('NThread',type="int",help='Number of OMP/BLAS/etc. threads to use. Default is %default ', default=1)
    OP.add_option('UpdateWeights',help='Update imaging weights. Default is %default',default="1")
    OP.add_option('DebugPdb',type="int",help='Drop into Pdb on error. Default is %default')

    # OP.OptionGroup("* PreApply Solution-related options","PreApply")
    # OP.add_option('PreApplySols')#,help='Solutions to apply to the data before solving.')
    # #OP.add_option('PreApplyMode',help='Solutions to apply to the data before solving.')

    OP.OptionGroup("* Solution-related options","Solutions")
    OP.add_option('ExtSols',type="str",help='External solution file. If set, will not solve.')
    OP.add_option('ApplyMode',type="str",help='Subtract selected sources. ')
    OP.add_option('ClipMethod',type="str",help='Clip data in the IMAGING_WEIGHT column. Can be set to Resid, DDEResid or ResidAnt . Default is %default')
    OP.add_option('OutSolsName',type="str",help='If specified will save the estimated solutions in this file. Default is %default')
    OP.add_option('ApplyToDir',type="int",help='Apply direction averaged gains to residual data in the mentioned direction. \
    If ApplyCal=-1 takes the mean gain over directions. -2 if off. Default is %default')
    OP.add_option('MergeBeamToAppliedSol',type="int",help='Use the beam in applied solution. Default is %default')
    OP.add_option('SkipExistingSols',type="int",help='Skipping existing solutions if they exist. Default is %default')
    OP.add_option('SolsDir',type="str",help='Directory in which to save the solutions. Default is %default')
    

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
    OP.add_option('PowerSmooth',type="float",help='When an antenna has missing baselines (like when using UVcuts) underweight its Q matrix. Default is %default') 
    OP.add_option('evPStep',type="int",help='Start calculation evP every evP_Step after that step. Default is %default')
    OP.add_option('evPStepStart',type="int",help='Calculate (I-KJ) matrix every evP_Step steps. Default is %default')
    OP.add_option('EvolutionSolFile',type="str",help='Evolution solution file. Default is %default')
    
    

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
    import killMS.Other.MyPickle
    killMS.Other.MyPickle.Save(OP,SaveFile)
    return OP
    

def main(OP=None,MSName=None):

    log.print("Checking system configuration:")
    # check for SHM size
    ram_size = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    shm_stats = os.statvfs('/dev/shm')
    shm_size = shm_stats.f_bsize * shm_stats.f_bavail
    shm_avail = shm_size / float(ram_size)

    if shm_avail < 0.6:
        log.print( ModColor.Str("""WARNING: max shared memory size is only {:.0%} of total RAM size.
            This can cause problems for large imaging jobs. A setting of 90% is recommended for 
            DDFacet and killMS. If your processes keep failing with SIGBUS or "bus error" messages,
            it is most likely for this reason. You can change the memory size by running
                $ sudo mount -o remount,size=90% /dev/shm
            To make the change permanent, edit /etc/defaults/tmps, and add a line saying "SHM_SIZE=90%".
            """.format(shm_avail)))
    else:
        log.print( "  Max shared memory size is {:.0%} of total RAM size".format(shm_avail))

    try:
        output = subprocess.check_output(["/sbin/sysctl", "vm.max_map_count"],universal_newlines=True)
    except Exception:
        log.print( ModColor.Str("""WARNING: /sbin/sysctl vm.max_map_count failed. Unable to check this setting."""))
        max_map_count = None
    else:
        max_map_count = int(output.strip().rsplit(" ", 1)[-1])

    if max_map_count is not None:
        if max_map_count < 500000:
            log.print( ModColor.Str("""WARNING: sysctl vm.max_map_count = {}. 
            This may be too little for large DDFacet and killMS jobs. If you get strange "file exists" 
            errors on /dev/shm, them try to bribe, beg or threaten your friendly local sysadmin into 
            setting vm.max_map_count=1000000 in /etc/sysctl.conf.
                """.format(max_map_count)))
        else:
            log.print( "  sysctl vm.max_map_count = {}".format(max_map_count))

    # check for memory lock limits
    import resource
    msoft, mhard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if msoft >=0 or mhard >=0:
        log.print(ModColor.Str("""WARNING: your system has a limit on memory locks configured.
            This may possibly slow down killMS performance. You can try removing the limit by running
                $ ulimit -l unlimited
            If this gives an "operation not permitted" error, you can try to bribe, beg or threaten 
            your friendly local sysadmin into doing
                # echo "*        -   memlock     unlimited" >> /etc/security/limits.conf
        """))


    if OP==None:
        import killMS.Other.MyPickle
        OP = killMS.Other.MyPickle.Load(SaveFile)

        
    options=OP.GiveOptionObject()

    # OMS: crude hack for now, can be removed when the more sophisticed DDF Parset is ported here.
    # I need option type info in order to generate a stimela schema, but this is not defined in the parset.
    # The type information is only available from the OP (class MyOptParse) object constructed directly
    # in the code above.
    # So, as a hack: "kMS.py --MSName MAKE_SCHEMA" will generate the schema file here.
    # If the DDFacet Parset class is ported to kMS, then this can be replaced by the DDF-style standalone
    # generate_stimela_schema.py script

    if options.MSName == "MAKE_SCHEMA":
        import killMS.Parset.generate_stimela_schema 
        output_name = os.path.dirname(killMS.Parset.generate_stimela_schema.__file__) + "/killms_stimela_schema.yaml"
        killMS.Parset.generate_stimela_schema.generate_schema(OP.parameter_types, output_name)
        sys.exit(0)


    # ## I've carefully moved the import statements around so that numpy is not yet imported at this
    # ## point. This gives us a chance to set the OPENBLAS thread variables and such.
    # ## But in case of someone messing around with imports in the future, leave this check here
    # if 'numpy' in sys.modules:
    #     raise RuntimeError("numpy already imported. This is a bug -- it shouldn't be imported yet")
    # os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OPENBLAS_MAX_THREADS'] = str(options.NThread)

    # now do all the other imports

    # from killMS.Data import MergeJones
    from killMS.Array import NpShared
    from killMS.Data import ClassJonesDomains
    import time
    import numpy as np
    import pickle
    from SkyModel.Sky import ClassSM
    from killMS.Wirtinger.ClassWirtingerSolver import ClassWirtingerSolver

    from killMS.Data import ClassVisServer
    from DDFacet.Data import ClassVisServer as ClassVisServer_DDF

    from killMS.Predict.PredictGaussPoints_NumExpr5 import ClassPredictParallel as ClassPredict

    from killMS.Array import ModLinAlg
    from killMS.Array import NpShared
    from killMS.Other import reformat

    #IdSharedMem=str(int(np.random.rand(1)[0]*100000))+"."
    global IdSharedMem
    IdSharedMem=str(int(os.getpid()))+"."
    DoApplyCal=(options.ApplyToDir!=-2)
    if type(options.ClipMethod)!=list:
        raise ValueError("Clipmethod is expected to be a list")

    ReWeight=(len(options.ClipMethod)>0)


    if MSName!=None:
        options.MSName=MSName

    if options.MSName=="":
        print("Give an MS name!")
        exit()

    # if options.SkyModel=="":
    #     print "Give a Sky Model!"
    #     exit()
    # if not(".npy" in options.SkyModel):
    #     print "Give a numpy sky model!"
    #     exit()

    TChunk=float(options.TChunk)
    dt=float(options.dt)

    if dt > TChunk*60:
        log.print(ModColor.Str("dt=%.2fm larger than TChunk. Setting dt=%.2fm"%(dt, TChunk*60)))
        dt = TChunk*60

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
    DicoSelectOptions["UVRangeKm"]=None
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


    if options.SolsDir is None:
        ParsetName="%skillMS.%s.sols.parset"%(reformat.reformat(options.MSName),SolsName)
    else:
        _MSName=reformat.reformat(options.MSName).split("/")[-2]
        DirName="%s%s"%(reformat.reformat(options.SolsDir),_MSName)
        if not os.path.isdir(DirName):
            os.makedirs(DirName)
        ParsetName="%s/killMS.%s.sols.parset"%(DirName,SolsName)
    OP.ToParset(ParsetName)





    APP=None
    GD=OP.DicoConfig
    if GD["SkyModel"]["SkyModel"]!="":
        log.print(ModColor.Str("Predict Mode: Catalog"))
        PredictMode="Catalog"
    elif GD["SkyModel"]["SkyModelCol"] is not None:
        log.print(ModColor.Str("Predict Mode: using column %s"%options.SkyModelCol))
        PredictMode="Column"
    else:
        log.print(ModColor.Str("Predict Mode: Image"))
        PredictMode="Image"
        BaseImageName=GD["ImageSkyModel"]["BaseImageName"]

        # ParsetName=GD["ImageSkyModel"]["ImagePredictParset"]
        # if ParsetName=="":
        #     ParsetName="%s.parset"%BaseImageName
        # log.print("Predict Mode: Image, with Parset: %s"%ParsetName)
        # GDPredict=ReadCFG.Parset(ParsetName).DicoPars

        if options.DicoModel!="" and options.DicoModel is not None:
            FileDicoModel=options.DicoModel
        else:
            FileDicoModel="%s.DicoModel"%BaseImageName


            
        ## OMS: only import it here, because otherwise is pulls in numpy too early, before I can fix
        ## the OPENBLAS threads thing
        import DDFacet.Other.MyPickle
        log.print("Reading model file %s"%FileDicoModel)
        GDPredict=DDFacet.Other.MyPickle.Load(FileDicoModel)["GD"]
        GDPredict["Output"]["Mode"] = "Predict"

        # if options.BeamModel is not None and options.BeamModel.lower()=="same":
        #     log.print(ModColor.Str("Setting kMS beam model from DDF parset..."))
        #     GD["Beam"]['BeamModel']=options.BeamModel=GDPredict["Beam"]["Model"]
        #     GD["Beam"]['NChanBeamPerMS']=options.NChanBeamPerMS=GDPredict["Beam"]["NBand"]
        #     GD["Beam"]['BeamAt']=options.BeamAt = GDPredict["Beam"]["At"] # tessel/facet
        #     GD["Beam"]['LOFARBeamMode']=options.LOFARBeamMode = GDPredict["Beam"]["LOFARBeamMode"]     # A/AE
        #     GD["Beam"]['DtBeamMin']=options.DtBeamMin = GDPredict["Beam"]["DtBeamMin"]
        #     GD["Beam"]['CenterNorm']=options.CenterNorm = GDPredict["Beam"]["CenterNorm"]
        #     GD["Beam"]['FITSFile']=options.FITSFile = GDPredict["Beam"]["FITSFile"]
        #     GD["Beam"]['FITSParAngleIncDeg']=options.FITSParAngleIncDeg = GDPredict["Beam"]["FITSParAngleIncDeg"]
        #     GD["Beam"]['FITSLAxis']=options.FITSLAxis        = GDPredict["Beam"]["FITSLAxis"]
        #     GD["Beam"]['FITSMAxis']=options.FITSMAxis        = GDPredict["Beam"]["FITSMAxis"]
        #     GD["Beam"]['FITSFeed']=options.FITSFeed	 = GDPredict["Beam"]["FITSFeed"] 
        #     GD["Beam"]['FITSVerbosity']=options.FITSVerbosity	 = GDPredict["Beam"]["FITSVerbosity"]
        #     GD["Beam"]["FeedAngle"]=options.FeedAngle	 = GDPredict["Beam"]["FeedAngle"]
        #     GD["Beam"]["ApplyPJones"]=options.ApplyPJones             = GDPredict["Beam"]["ApplyPJones"]
        #     GD["Beam"]["FlipVisibilityHands"]=options.FlipVisibilityHands     = GDPredict["Beam"]["FlipVisibilityHands"]
        #     GD["Beam"]['FITSFeedSwap']=options.FITSFeedSwap=GDPredict["Beam"]["FITSFeedSwap"]
            
            
            
            


        
        if not "StokesResidues" in GDPredict["Output"].keys():
            log.print(ModColor.Str("Seems like the DicoModel was built by an older version of DDF"))
            log.print(ModColor.Str("   ... updating keywords"))
            GDPredict["Output"]["StokesResidues"]="I"

        GDPredict["Data"]["MS"]=options.MSName
        if options.DDFCacheDir!='':
            GDPredict["Cache"]["Dir"]=options.DDFCacheDir

        if not("PSFFacets" in GDPredict["RIME"].keys()):
               GDPredict["RIME"]["PSFFacets"]=0
               GDPredict["RIME"]["PSFOversize"]=1

        GDPredict["Beam"]["NBand"]=options.NChanBeamPerMS
        GDPredict["Freq"]["NDegridBand"]=options.NChanSols
        #GDPredict["Compression"]["CompDeGridMode"]=False
        #GDPredict["Compression"]["CompDeGridMode"]=True
        GDPredict["RIME"]["ForwardMode"]="Classic"
        GDPredict["Cache"]["CF"]=False

        if options.ChanSlice is not None:
            GDPredict["Selection"]["ChanStart"]=int(options.ChanSlice[0])
            GDPredict["Selection"]["ChanEnd"]=int(options.ChanSlice[1])
            GDPredict["Selection"]["ChanStep"]=int(options.ChanSlice[2])

        #GDPredict["Caching"]["ResetCache"]=1
        if options.MaxFacetSize:
            GDPredict["Facets"]["DiamMax"]=options.MaxFacetSize
        else:
            OP.options.ImageSkyModel_MaxFacetSize=OP.DicoConfig["ImageSkyModel"]["MaxFacetSize"]=GDPredict["Facets"]["DiamMax"]

        if options.MinFacetSize:
            GDPredict["Facets"]["DiamMin"]=options.MinFacetSize
        else:
            OP.options.ImageSkyModel_MinFacetSize=OP.DicoConfig["ImageSkyModel"]["MinFacetSize"]=GDPredict["Facets"]["DiamMin"]

        if options.Decorrelation is not None and options.Decorrelation != "":
            log.print(ModColor.Str("Overwriting DDF parset decorrelation mode [%s] with kMS option [%s]"\
                                    %(GDPredict["RIME"]["DecorrMode"],options.Decorrelation)))
            GDPredict["RIME"]["DecorrMode"]=options.Decorrelation
        else:
            
            GD["SkyModel"]["Decorrelation"]=DoSmearing=options.Decorrelation=GDPredict["RIME"]["DecorrMode"]
            OP.options.SkyModel_Decorrelation=options.Decorrelation
            log.print(ModColor.Str("Decorrelation mode will be [%s]" % DoSmearing))

        OP.ToParset(ParsetName)

        # if options.Decorrelation != GDPredict["DDESolutions"]["DecorrMode"]:
        #     log.print(ModColor.Str("Decorrelation modes for DDFacet and killMS are different [%s vs %s respectively]"\)
        #                             %(GDPredict["DDESolutions"]["DecorrMode"],options.Decorrelation))
        # GDPredict["DDESolutions"]["DecorrMode"]=options.Decorrelation
        
        if options.OverS is not None:
            GDPredict["CF"]["OverS"]=options.OverS
        if options.wmax is not None:
            GDPredict["CF"]["wmax"]=options.wmax

        GDPredict["Facets"].setdefault("MixingWidth", 10)  # for compatibility with older DicoModels

        GD["GDImage"]=GDPredict
        GDPredict["GDkMS"]=GD

        from DDFacet.Other import AsyncProcessPool
        from DDFacet.Other import Multiprocessing
        AsyncProcessPool._init_default()
        APP=AsyncProcessPool.APP
        AsyncProcessPool.init(ncpu=NCPU, affinity=GDPredict["Parallel"]["Affinity"],
                              verbose=GDPredict["Debug"]["APPVerbose"])
        VS_DDFacet=ClassVisServer_DDF.ClassVisServer(options.MSName,
                                                     ColName=GDPredict["Data"]["ColName"],
                                                     #TVisSizeMin=GDPredict["Data"]["ChunkHours"]*60,
                                                     #DicoSelectOptions=DicoSelectOptions,
                                                     TChunkSize=GDPredict["Data"]["ChunkHours"],
                                                     #IdSharedMem=IdSharedMem,
                                                     #Robust=GDPredict["Weight"]["Robust"],
                                                     #Weighting=GDPredict["Weight"]["Type"],
                                                     #MFSWeighting=GDPredict["Weight"]["MFSWeighting"],
                                                     #Super=GDPredict["Weight"]["Super"],
                                                     #DicoSelectOptions=dict(GDPredict["Selection"]),
                                                     #NCPU=GDPredict["Parallel"]["NCPU"],
                                                     GD=GDPredict)


        


    #SM.SourceCat.I*=1000**2
    VS=ClassVisServer.ClassVisServer(options.MSName,ColName=ReadColName,
                                     TVisSizeMin=dt,
                                     DicoSelectOptions=DicoSelectOptions,
                                     TChunkSize=TChunk,IdSharedMem=IdSharedMem,
                                     NCPU=NCPU,
                                     Weighting=options.Weighting,
                                     Robust=options.Robust,
                                     WeightUVMinMax=options.WeightUVMinMax,
                                     WTUV=options.WTUV,
                                     GD=GD)

    print(VS.MS)
    if not(WriteColName in VS.MS.ColNames) and \
            WriteColName is not None and \
            WriteColName != "None" and \
            WriteColName != "":
        log.print( "Column %s not in MS "%WriteColName)
        VS.MS.AddCol(WriteColName,LikeCol="DATA")
        #exit()
    if not(ReadColName in VS.MS.ColNames):
        log.print( "Column %s not in MS "%ReadColName)
        exit()

    VS_PredictCol=None
    if PredictMode=="Catalog":
        SM=ClassSM.ClassSM(options.SkyModel,
                           killdirs=kills,
                           invert=invert)
        SM.Type="Catalog"
        Alpha=SM.SourceCat.alpha
        Alpha[np.isnan(Alpha)]=0
    elif PredictMode=="Image":
        from killMS.Predict import ClassImageSM2 as ClassImageSM
        #from killMS.Predict import ClassImageSM3 as ClassImageSM
        
        PreparePredict=ClassImageSM.ClassPreparePredict(BaseImageName,VS_DDFacet,IdSharedMem,GD=GDPredict)#,IdSharedMem=IdSharedMem)
        SM=PreparePredict.SM
        #VS.setGridProps(PreparePredict.FacetMachine.Cell,PreparePredict.FacetMachine.NpixPaddedFacet)
        VS.setGridProps(PreparePredict.FacetMachine.Cell,None)#PreparePredict.FacetMachine.NpixPaddedFacet)
        FacetMachine=PreparePredict.FacetMachine
        VS.setFOV(FacetMachine.OutImShape,FacetMachine.PaddedGridShape,FacetMachine.FacetShape,FacetMachine.CellSizeRad)
    elif PredictMode=="Column":
        VS_PredictCol=ClassVisServer.ClassVisServer(options.MSName,ColName=GD["SkyModel"]["SkyModelCol"],
                                                    TVisSizeMin=dt,
                                                    DicoSelectOptions=DicoSelectOptions,
                                                    TChunkSize=TChunk,IdSharedMem=IdSharedMem+"ColPredict.",
                                                    NCPU=NCPU,
                                                    Weighting=options.Weighting,
                                                    Robust=options.Robust,
                                                    WeightUVMinMax=options.WeightUVMinMax,
                                                    WTUV=options.WTUV,
                                                    GD=GD)
        class cSM:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        ClusterCat=np.zeros((1,),dtype=[('Name', 'S200'), ('ra', '<f8'), ('dec', '<f8'), ('SumI', '<f8'), ('Cluster', '<i8')]).view(np.recarray)
        ClusterCat.ra=VS.MS.rac
        ClusterCat.dec=VS.MS.decc
        ClusterCat.SumI=1.
        ClusterCat.Cluster=0
        SM=cSM(Type="Column",NDir=1,ClusterCat=ClusterCat)
        VS_PredictCol.setSM(SM)
        VS_PredictCol.CalcWeigths()

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
                           "DoReg":False,
                           "gamma":1,
                           "AmpQx":.5,
                           "PrecisionDot":options.PrecisionDot,
                           "DicoMergeStations":VS.DicoMergeStations}

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
                                VS_PredictCol=VS_PredictCol,
                                GD=GD)
    
    
    Solver.InitSol(TestMode=False)

    PM=ClassPredict(NCPU=NCPU,IdMemShared=IdSharedMem,DoSmearing=DoSmearing,BeamAtFacet=(GD["Beam"]["BeamAt"].lower() == "facet"))
    PM2=None#ClassPredict_orig(NCPU=NCPU,IdMemShared=IdSharedMem)


    if (options.SolverType=="KAFCA"):

        Solver.InitMeanBeam()
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

    # #with warnings.catch_warnings():
    # #    warnings.filterwarnings('error')
    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    # # ##############################

    iChunk=0

    while True:

        Load=VS.LoadNextVisChunk()
        if SM.Type=="Column":
            VS_PredictCol.LoadNextVisChunk()
        if Load=="EndOfObservation":
            break
        if Load == "Empty":
            Solver.AppendGToSolArray()
            log.print( "skipping rest of processing for this chunk")
            continue


        iChunk+=1
        #if iChunk<6: continue
        if options.ExtSols=="":
            SaveSols=True
            if options.SubOnly==0:
                if options.Parallel:
                    Solver.doNextTimeSolve_Parallel(Parallel=True)
                    #Solver.doNextTimeSolve_Parallel(Parallel=True,
                    #                                SkipMode=True)
                else:
                    #Solver.doNextTimeSolve_Parallel(SkipMode=True)
                    Solver.doNextTimeSolve()#SkipMode=True)
            else:
                DoSubstract=1

            
            def SavePredict(ArrayName,FullPredictColName):
                log.print( "Writing full predicted data in column %s of %s"%(FullPredictColName,options.MSName))
                VS.MS.AddCol(FullPredictColName)
                PredictData=NpShared.GiveArray("%s%s"%(IdSharedMem,ArrayName))
                t=VS.MS.GiveMainTable(readonly=False)#table(VS.MS.MSName,readonly=False,ack=False)
                nrow_ThisChunk=Solver.VS.MS.ROW1-Solver.VS.MS.ROW0
                d=np.zeros((nrow_ThisChunk,VS.MS.NChanOrig,4),PredictData.dtype)
                d[:,VS.MS.ChanSlice,:]=VS.MS.ToOrigFreqOrder(PredictData)
                if Solver.VS.MS.NPolOrig==2:
                    dc=d[:,:,0::3]
                else:
                    dc=d
                t.putcol(FullPredictColName,dc,Solver.VS.MS.ROW0,nrow_ThisChunk)
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
                if WriteColName is not None and WriteColName != "None" and WriteColName != "":
                    log.print( "Subtracting free predict from data")
                    PredictData=NpShared.GiveArray("%s%s"%(IdSharedMem,"PredictedDataGains"))
                    Solver.VS.ThisDataChunk["data"]-=PredictData
                    log.print( "  save visibilities in %s column"%WriteColName)
                    t=Solver.VS.MS.GiveMainTable(readonly=False)#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                    d=VS.MS.ToOrigFreqOrder(Solver.VS.MS.data)
                    if Solver.VS.MS.NPolOrig==2:
                        dc=d[:,:,0::3]
                    else:
                        dc=d
                    t.putcol(WriteColName,dc,Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                    t.close()
                else:
                    log.print("No output column specified. Skipping writing visibility data.")

            Sols=Solver.GiveSols(SaveStats=True)
            
            # ##########
            # FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),SolsName)

            # log.print( "Save Solutions in file: %s"%FileName)
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


                if options.SolsDir is None:
                    FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),SolsName)
                else:
                    _MSName=reformat.reformat(options.MSName).split("/")[-2]
                    DirName=os.path.abspath("%s%s"%(reformat.reformat(options.SolsDir),_MSName))
                    if not os.path.isdir(DirName):
                        os.makedirs(DirName)
                    FileName="%s/killMS.%s.sols.npz"%(DirName,SolsName)
                    

                log.print( "Save Solutions in file: %s"%FileName)
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
                         MSName=os.path.abspath(VS.MS.MSName),
                         MSNameTime0=VS.MS.Time0,
                         Sols=SolsSave,
                         StationNames=StationNames,
                         SkyModel=ClusterCat,
                         ClusterCat=ClusterCat,
                         SourceCatSub=SourceCatSub,
                         ModelName=options.SkyModel,
                         FreqDomains=VS.SolsFreqDomains,
                         BeamTimes=VS.BeamTimes)

                # RA,DEC=ClusterCat.ra,ClusterCat.dec
                # from killMS.Other.rad2hmsdms import rad2hmsdms
                # for i in range(RA.size): 
                #     ra,dec=RA[i],DEC[i]
                #     print rad2hmsdms(ra,Type="ra").replace(" ",":"),rad2hmsdms(dec,Type="dec").replace(" ",".")


        else:

            ExtSolsName=options.ExtSols
            if options.SolsDir is None:
                FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),ExtSolsName)
            else:
                _MSName=reformat.reformat(options.MSName).split("/")[-2]
                DirName=os.path.abspath("%s%s"%(reformat.reformat(options.SolsDir),_MSName))
                if not os.path.isdir(DirName):
                    os.makedirs(DirName)
                FileName="%s/killMS.%s.sols.npz"%(DirName,ExtSolsName)

            log.print("Loading external solution file: %s"%FileName)
            DicoLoad=np.load(FileName)
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
            G=np.require(G, dtype=np.complex64, requirements="C")


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


            JonesMerged=Jones

            if options.BeamModel==None or not options.MergeBeamToAppliedSol:
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

            if ("Resid" in options.ClipMethod) or ("DDEResid" in options.ClipMethod):
                log.print( ModColor.Str("Clipping bad solution-based data ... ",col="green"))
                

                nrows=Solver.VS.ThisDataChunk["times"].size

                Solver.VS.ThisDataChunk["W"]=np.ones((nrows,Solver.VS.MS.ChanFreq.size),np.float64)


                ################
                # #PM.GiveCovariance(Solver.VS.ThisDataChunk,Jones)

                log.print("   Compute residual data")
                Predict=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=JonesMerged)
                Solver.VS.ThisDataChunk["resid"]=Solver.VS.ThisDataChunk["data"]-Predict
                Weights=Solver.VS.ThisDataChunk["W"]


                if "Resid" in options.ClipMethod:
                    Diff=Solver.VS.ThisDataChunk["resid"]
                    std=np.std(Diff[Solver.VS.ThisDataChunk["flags"]==0])
                    log.print( "   Estimated standard deviation in the residual data: %f"%std)

                    ThresHold=5.
                    cond=(np.abs(Diff)>ThresHold*std)
                    ind=np.any(cond,axis=2)
                    Weights[ind]=0.

                if "DDEResid" in options.ClipMethod:
                    log.print("   Compute corrected residual data in all directions")
                    PM.GiveCovariance(Solver.VS.ThisDataChunk,JonesMerged,SM)



                Weights=Solver.VS.ThisDataChunk["W"]
                NNotFlagged=np.count_nonzero(Weights)
                log.print("   Set weights to Zero for %5.2f %% of data"%(100*float(Weights.size-NNotFlagged)/(Weights.size)))

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

                if options.UpdateWeights:
                    log.print( "  Writing in IMAGING_WEIGHT column ")
                    VS.MS.AddCol("IMAGING_WEIGHT",ColDesc="IMAGING_WEIGHT")
                    t=Solver.VS.MS.GiveMainTable(readonly=(options.UpdateWeights==0)) # table(Solver.VS.MS.MSName,readonly=False,ack=False)
                    WAllChans=t.getcol("IMAGING_WEIGHT",Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                    WAllChans[:,Solver.VS.MS.ChanSlice]=VS.MS.ToOrigFreqOrder(Weights)
                    t.putcol("IMAGING_WEIGHT",WAllChans,Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                    t.close()
                else:
                    log.print("  Imaging weight update not requested, skipping")
                    WallChans=None


            if "ResidAnt" in options.ClipMethod and options.SubOnly==0:
                log.print("Compute weighting based on antenna-selected residual")
                DomainMachine.AddVisToJonesMapping(Jones,times,freqs)
                nrows=Solver.VS.ThisDataChunk["times"].size
                Solver.VS.ThisDataChunk["W"]=np.ones((nrows,Solver.VS.MS.ChanFreq.size),np.float64)
                PM.GiveCovariance(Solver.VS.ThisDataChunk,Jones,SM,Mode="ResidAntCovariance")

                Weights=Solver.VS.ThisDataChunk["W"]
                # Weights/=np.mean(Weights)
                
                if options.UpdateWeights:
                    log.print( "  Writing in IMAGING_WEIGHT column ")
                    VS.MS.AddCol("IMAGING_WEIGHT",ColDesc="IMAGING_WEIGHT")
                    t=Solver.VS.MS.GiveMainTable(readonly=(options.UpdateWeights==0))#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                    WAllChans=t.getcol("IMAGING_WEIGHT",Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                    WAllChans[:,Solver.VS.MS.ChanSlice]=VS.MS.ToOrigFreqOrder(Weights)

                    Med=np.median(WAllChans)
                    Sig=1.4826*np.median(np.abs(WAllChans-Med))
                    Cut=Med+5*Sig
                    WAllChans[WAllChans>Cut]=Cut
                    t.putcol("IMAGING_WEIGHT",WAllChans,Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                    t.close()
                else:
                    log.print("  Imaging weight update not requested, skipping")
                    WAllChans=None

                if WAllChans is not None:
                    ID=Solver.VS.MS.ROW0
                    if options.SolsDir is None:
                        FileName="%skillMS.%s.Weights.%i.npy"%(reformat.reformat(options.MSName),SolsName,ID)
                    else:
                        _MSName=reformat.reformat(options.MSName).split("/")[-2]
                        DirName=os.path.abspath("%s%s"%(reformat.reformat(options.SolsDir),_MSName))
                        if not os.path.isdir(DirName):
                            os.makedirs(DirName)
                        FileName="%s/killMS.%s.Weights.%i.npy"%(DirName,SolsName,ID)
                    log.print( "  Saving weights in file %s"%FileName)
                    np.save(FileName,WAllChans)
                else:
                    log.print("  Weights not computed so not saving them")
                
            if DoSubstract:
                log.print( ModColor.Str("Subtract sources ... ",col="green"))
                if options.SubOnly==0:
                    SM.SelectSubCat(SM.SourceCat.kill==1)

                SourceCatSub=SM.SourceCat.copy()

                PredictData=PM.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=JonesMerged)

                # PredictColName=options.PredictColName
                # if PredictColName!="":
                #     log.print( "Writing predicted data in column %s of %s"%(PredictColName,MSName))
                #     VS.MS.AddCol(PredictColName)
                #     t=Solver.VS.MS.GiveMainTable(readonly=False)#table(VS.MS.MSName,readonly=False,ack=False)
                #     t.putcol(PredictColName,VS.MS.ToOrigFreqOrder(PredictData),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                #     t.close()
                    
                #PredictData2=PM2.predictKernelPolCluster(Solver.VS.ThisDataChunk,Solver.SM,ApplyTimeJones=Jones)
                #diff=(PredictData-PredictData2)
                #print diff
                #ind=np.where(diff==np.max(diff))
                #print ind
                #print np.max(PredictData-PredictData2)
                #print np.where(np.isnan(diff))
                #print PredictData[1997:1999],PredictData[1997:1999]

                Solver.VS.ThisDataChunk["data"]-=PredictData

                #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                #Solver.VS.ThisDataChunk["data"]=PredictData
                SM.RestoreCat()

            if DoApplyCal:
                log.print( ModColor.Str("Apply calibration in direction: %i"%options.ApplyToDir,col="green"))
                G=JonesMerged["Jones"]
                GH=JonesMerged["JonesH"]
                if not("A" in options.ApplyMode):
                    gabs=np.abs(G)
                    gabs[gabs==0]=1.
                    G/=gabs
                    GH/=gabs
                PM.ApplyCal(Solver.VS.ThisDataChunk,JonesMerged,options.ApplyToDir)

            Solver.VS.MS.data=Solver.VS.ThisDataChunk["data"]
            Solver.VS.MS.flags_all=Solver.VS.ThisDataChunk["flags"]
            # Solver.VS.MS.SaveVis(Col=WriteColName)

            if (DoSubstract|DoApplyCal):
                if WriteColName is not None and WriteColName != "None" and WriteColName != "":
                    log.print( "Save visibilities in %s column"%WriteColName)
                    t=Solver.VS.MS.GiveMainTable(readonly=False)#table(Solver.VS.MS.MSName,readonly=False,ack=False)
                    nrow_ThisChunk=Solver.VS.MS.ROW1-Solver.VS.MS.ROW0
                    d=np.zeros((nrow_ThisChunk,VS.MS.NChanOrig,4),Solver.VS.ThisDataChunk["data"].dtype)
                    d[:,VS.MS.ChanSlice,:]=VS.MS.ToOrigFreqOrder(Solver.VS.ThisDataChunk["data"])

                    if Solver.VS.MS.NPolOrig==2:
                        dc=d[:,:,0::3]
                    else:
                        dc=d
                    t.putcol(WriteColName,dc,Solver.VS.MS.ROW0,nrow_ThisChunk)
                    #t.putcol("FLAG",VS.MS.ToOrigFreqOrder(Solver.VS.MS.flags_all),Solver.VS.MS.ROW0,Solver.VS.MS.ROW1-Solver.VS.MS.ROW0)
                    t.close()
                else:
                    log.print("No output column specified. Skipping writing visibility data.")

                
    if APP is not None:
        #APP.terminate()
        APP.shutdown()
        del(APP)
        Multiprocessing.cleanupShm()

    NpShared.DelAll(IdSharedMem)

def GiveNoise(options,DicoSelectOptions,IdSharedMem,SM,PM,PM2,ConfigJacobianAntenna,GD):
    log.print( ModColor.Str("Initialising Kalman filter with Levenberg-Maquardt estimate"))
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
        log.print( "Antenna %s have abnormal noise (Numbers %s)"%(str(Stations[indFlag]),str(indFlag)))
    
    indTake=np.where((rmsAnt-Mean_rmsAnt)/Mean_rmsAnt<Thr)[0]
     
    gscale=np.mean(np.abs(G[:,:,indTake,:,0,0]))
    TrueMeanRMSAnt=np.mean(rmsAnt[indTake])
      
    GG=np.mean(np.mean(np.mean(np.abs(G[0,:]),axis=0),axis=0),axis=0)
    GGprod= np.dot( np.dot(GG,np.ones((2,2),float)*TrueMeanRMSAnt) , GG.T)
    rms=np.mean(GGprod)
    log.print( "Estimated rms: %f Jy"%(rms))
    return rms,SolverInit.G


def _exc_handler(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type is SyntaxError:
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"


    sys.excepthook = _exc_handler

def driver():
    from killMS.Other import logo
    from killMS.Other import ModColor
    from killMS.Other import MyPickle


    tic = time.time()
    #os.system('clear')

    logo.print_logo()
    if len(sys.argv)<2:
        raise RuntimeError('At least one parset name or option must be supplied')
        
    ParsetFile=sys.argv[1]

    TestParset=ReadCFG.Parset(ParsetFile)
    if TestParset.Success==True:
        #global Parset
        Parset=TestParset
        log.print(ModColor.Str("Successfully read %s parset"%ParsetFile))

    OP=read_options()
    options=OP.GiveOptionObject()

    if options.DoBar=="0":
        from DDFacet.Other.progressbar import ProgressBar
        ProgressBar.silent=1

    if options.DebugPdb==1:
        sys.excepthook = _exc_handler

    
    #main(OP=OP)

    import glob
    MSName=options.MSName

    if ".txt" in MSName:
        f=open(MSName)
        Ls=f.readlines()
        f.close()
        MSName=[]
        for l in Ls:
            ll=l.replace("\n","")
            MSName.append(ll)
        lMS=MSName
        log.print( "In batch mode, running killMS on the following MS:")
        for MS in lMS:
            log.print( "  %s"%MS)
    else:
        lMS=options.MSName

    BaseParset="BatchCurrentParset.parset"
    OP.ToParset(BaseParset)    
    import os
    try:
        if type(lMS)==list:
            for MSName in lMS:

                if options.SkipExistingSols:
                    SolsName=options.SolverType
                    if options.OutSolsName!="":
                        SolsName=options.OutSolsName
                    # FileName="%skillMS.%s.sols.npz"%(reformat.reformat(MSName),SolsName)

                    if options.SolsDir is None:
                        FileName="%skillMS.%s.sols.npz"%(reformat.reformat(MSName),SolsName)
                    else:
                        _MSName=reformat.reformat(MSName).split("/")[-2]
                        DirName=os.path.abspath("%s%s"%(reformat.reformat(options.SolsDir),_MSName))
                        if not os.path.isdir(DirName):
                            os.makedirs(DirName)
                        FileName="%s/killMS.%s.sols.npz"%(DirName,SolsName)

                    log.print("Checking %s"%FileName)
                    if os.path.isfile(FileName):
                        log.print(ModColor.Str("Solution file %s exist"%FileName))
                        log.print(ModColor.Str("   SKIPPING"))
                        continue

                ss="kMS.py %s --MSName=%s"%(BaseParset,MSName)
                log.print("Running %s"%ss)
                os.system(ss)
                if options.RemoveDDFCache:
                    os.system("rm -rf %s*ddfcache"%MSName)
        else:
            main(OP=OP,MSName=MSName)
        toc = time.time()     
        elapsed = toc - tic
        log.print( ModColor.Str("Dataset(s) calibrated successfully in " \
                                 "{0:02.0f}h{1:02.0f}m{2:02.0f}s".format(
                                 (elapsed // 60) // 60,
                                 (elapsed // 60) % 60,
                                 elapsed % 60)))
    except:
        # log.print( traceback.format_exc())
        if IdSharedMem is not None:
            from killMS.Array import NpShared
            NpShared.DelAll(IdSharedMem)
        raise

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()
