
import numpy as np
from DDFacet.Imager.ClassDeconvMachine import ClassImagerDeconv
from pyrap.images import image
from killMS2.Array import NpShared

from killMS2.Other import MyLogger
log=MyLogger.getLogger("ClassImageSM")
from killMS2.Other.progressbar import ProgressBar
from DDFacet.ToolsDir.GiveEdges import GiveEdges

class ClassImageSM():
    def __init__(self):
        self.Type="Image"


class ClassPreparePredict(ClassImagerDeconv):

    def __init__(self,ModelImageName,VS,*args,**kwargs):
        ClassImagerDeconv.__init__(self,**kwargs)
        self.ModelImageName=ModelImageName
        self.VS=VS
        self.IdSharedMem=kwargs["IdSharedMem"]
        self.SM=ClassImageSM()
        self.InitFacetMachine()
        self.LoadModel()

    def LoadModel(self):
        im=image(self.ModelImageName)
        data=im.getdata()

        nch,npol,_,_=data.shape
        for ch in range(nch):
            for pol in range(npol):
                data[ch,pol]=data[ch,pol].T[::-1]
        self.ModelImage=data
        
        self.ModelImage=NpShared.ToShared("%sModelImage"%(self.IdSharedMem),self.ModelImage)
        del(data)
        self.DicoImager=self.FacetMachine.DicoImager
        
        NFacets=len(self.FacetMachine.DicoImager)
        self.NDirs=NFacets
        self.Dirs=range(self.NDirs)
        ClusterCat=np.zeros((len(self.Dirs),),dtype=[('Name','|S200'),
                                                     ('ra',np.float),('dec',np.float),
                                                     ('l',np.float),('m',np.float),
                                                     ('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)
        self.DicoImager=self.FacetMachine.DicoImager
        self.ClusterCat=ClusterCat

        print>>log, "Splitting model image"
        self.BuildGridsParallel()

        #ind=np.where(self.ClusterCat.SumI!=0)[0]
        #self.ClusterCat=self.ClusterCat[ind].copy()
        #NFacets=self.ClusterCat.shape[0]
        #print>>log, "  There are %i non-zero facets"%NFacets



        #self.BuildGridsSerial()
        #self.BuildGridsParallel()

        NFacets=self.ClusterCat.shape[0]
        self.SM.NDir=NFacets
        self.SM.Dirs=self.Dirs
        self.SM.ClusterCat=self.ClusterCat
        self.SM.GD=self.FacetMachine.GD
        self.SM.DicoImager=self.FacetMachine.DicoImager
        self.SM.GD["Compression"]["CompDeGridMode"]=0
        self.SM.rac=self.VS.MS.rac
        self.SM.decc=self.VS.MS.decc
        
        





    def BuildGridsParallel(self):
        print>>log, "Building the grids"
        ListGrid=[]



        NCPU=self.GD["Parallel"]["NCPU"]

        NFacets=len(self.DicoImager.keys())
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        NJobs=NFacets
        for iFacet in range(NFacets):
            work_queue.put(iFacet)

        GM=self.FacetMachine.GiveGM(0)
        argsImToGrid=(GM.GridShape,GM.PaddingInnerCoord,GM.OverS,GM.Padding,GM.dtype)

        NormImage=self.FacetMachine.GiveNormImage()
        _=NpShared.ToShared("%sNormImage"%self.IdSharedMem,NormImage)
        
        workerlist=[]
        for ii in range(NCPU):
            W=Worker(work_queue, result_queue,argsImToGrid=argsImToGrid,
                     IdSharedMem=self.IdSharedMem,
                     DicoImager=self.FacetMachine.DicoImager)
            workerlist.append(W)
            workerlist[ii].start()


        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Make Grids ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NFacets))
        iResult=0

        ClusterCat=self.ClusterCat
        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
                iFacet=DicoResult["iFacet"]
                    
                ClusterCat.SumI[iFacet]=np.real(DicoResult["SumFlux"])
                ra,dec=self.FacetMachine.DicoImager[iFacet]["RaDec"]
                l0,m0=self.FacetMachine.DicoImager[iFacet]["l0m0"]
                ClusterCat.l[iFacet]=l0
                ClusterCat.m[iFacet]=m0

                ClusterCat.ra[iFacet]=ra
                ClusterCat.dec[iFacet]=dec

            NDone=iResult
            intPercent=int(100*  NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))


        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()


        DelFacet=(ClusterCat.SumI==0)

        D={}
        
        iFacetNew=0
        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            if DelFacet[iFacet]==0:
                Grid=NpShared.GiveArray("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet))
                ListGrid.append(Grid)
                D[iFacetNew]=self.FacetMachine.DicoImager[iFacet]
                iFacetNew+=1

        self.FacetMachine.DicoImager=D
        self.DicoImager=D
        self.ClusterCat=ClusterCat[DelFacet==0].copy()
        NFacets=self.ClusterCat.shape[0]
        self.ClusterCat.Cluster=np.arange(NFacets)
        self.Dirs=self.ClusterCat.Cluster.tolist()
        
            
        NpShared.PackListArray("%sGrids"%(self.IdSharedMem),ListGrid)
        NpShared.DelAll("%sModelFacet"%self.IdSharedMem)
        NpShared.DelAll("%sModelGrid"%self.IdSharedMem)
        return True


        



from DDFacet.Imager.ClassImToGrid import ClassImToGrid
 
import multiprocessing
from killMS2.Predict.PredictGaussPoints_NumExpr5 import ClassPredict
class Worker(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 argsImToGrid=None,
                 IdSharedMem=None,
                 DicoImager=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.IdSharedMem=IdSharedMem
        self.DicoImager=DicoImager
        self.SharedMemNameSphe="%sSpheroidal"%(self.IdSharedMem)
        self.ifzfCF=NpShared.GiveArray(self.SharedMemNameSphe)
        self.ClassImToGrid=ClassImToGrid(*argsImToGrid,ifzfCF=self.ifzfCF)
        self.Image=NpShared.GiveArray("%sModelImage"%(self.IdSharedMem))

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            try:
                iFacet = self.work_queue.get()
            except:
                break


            # ModelFacet=NpShared.GiveArray("%sModelFacet.%3.3i"%(self.IdSharedMem,iFacet))
            # Grid=self.ClassImToGrid.setModelIm(ModelFacet)
            # _=NpShared.ToShared("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet),Grid)
            # self.result_queue.put({"Success":True})

            Image=NpShared.GiveArray("%sModelImage"%(self.IdSharedMem))
            Grid,SumFlux=self.ClassImToGrid.GiveGridSharp(Image,self.DicoImager,iFacet)
            
            #NormImage=NpShared.GiveArray("%sNormImage"%self.IdSharedMem)
            #Grid,SumFlux=self.ClassImToGrid.GiveGridFader(Image,self.DicoImager,iFacet,NormImage)
            _=NpShared.ToShared("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet),Grid)
            self.result_queue.put({"Success":True,"iFacet":iFacet,"SumFlux":SumFlux})
            
