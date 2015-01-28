import numpy as np
from pyrap.tables import table
from Data.ClassMS import ClassMS
from Sky.ClassSM import ClassSM
from Other.ClassTimeIt import ClassTimeIt
import numexpr as ne
from Other.progressbar import ProgressBar
import multiprocessing
from Array import ModLinAlg
from Array import NpShared
from Data import ClassVisServer

import Sky.PredictGaussPoints_NumExpr2
import Sky.PredictGaussPoints_NumExpr

def test():
    MSName="/media/tasse/data/killMS2/TEST/Simul/0000.MS"
    SMName="/media/tasse/data/killMS2/TEST/Simul/ModelRandom00.txt.npy"
    ReadColName="DATA"
    WriteColName="DATA"
    NCPU=6
    Noise=100
    IdSharedMem=str(int(np.random.rand(1)[0]*100000))+"."

    SM=ClassSM(SMName)

    VS=ClassVisServer.ClassVisServer(MSName,ColName=ReadColName,
                                     TVisSizeMin=1,
                                     TChunkSize=14)
    Load=VS.LoadNextVisChunk()

    MS=VS.MS
    SM.Calc_LM(MS.rac,MS.decc)
    print MS

    MS.PutBackupCol(incol="CORRECTED_DATA")

    na=MS.na
    nd=SM.NDir
    NSols=MS.F_ntimes
    Sols=np.zeros((NSols,),dtype=[("t0",np.float64),("t1",np.float64),("tm",np.float64),("G",np.complex64,(na,nd,2,2))])
    Sols=Sols.view(np.recarray)
    Sols.G[:,:,:,0,0]=1
    Sols.G[:,:,:,1,1]=1
    Sols.G+=np.random.randn(*Sols.G.shape)+1j*np.random.randn(*Sols.G.shape)
    dt=MS.dt
    Sols.t0=MS.F_times-dt/2.
    Sols.t1=MS.F_times+dt/2.
    Sols.tm=MS.F_times
    
    Jones=Sky.PredictGaussPoints_NumExpr2.SolsToDicoJones(Sols,VS.MS.NSPWChan)

    PM=Sky.PredictGaussPoints_NumExpr2.ClassPredictParallel(NCPU=1)
    PredictData_p=PM.predictKernelPolCluster(VS.ThisDataChunk,SM,ApplyTimeJones=Jones)

    d0=VS.ThisDataChunk["data"].copy()
    PM.ApplyCal(VS.ThisDataChunk,Jones,0)
    dc0=VS.ThisDataChunk["data"].copy()

    PM=Sky.PredictGaussPoints_NumExpr2.ClassPredict(NCPU=6)
    PredictData=PM.predictKernelPolCluster(VS.ThisDataChunk,SM,ApplyTimeJones=Jones)

    print np.allclose(PredictData_p,PredictData)

    PM=Sky.PredictGaussPoints_NumExpr.ClassPredict(NCPU=6)
    PredictData0=PM.predictKernelPolCluster(VS.ThisDataChunk,SM,ApplyTimeJones=Jones)

    VS.ThisDataChunk["data"]=d0
    PM.ApplyCal(VS.ThisDataChunk,Jones,0)

    dc1=VS.ThisDataChunk["data"].copy()
    print np.allclose(dc0,dc1)

    print np.allclose(PredictData_p,PredictData0)

    stop

    # PM=ClassPredictParallel(NCPU=6)
    # SM.SelectSubCat(SM.SourceCat.kill==1)
    # PredictData=PM.predictKernelPolCluster(VS.ThisDataChunk,SM)#,ApplyTimeJones=Jones)
    # SM.RestoreCat()
