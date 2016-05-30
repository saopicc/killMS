
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

        # NormImage=self.FacetMachine.GiveNormImage()
        # import pylab
        # pylab.clf()
        # pylab.imshow(NormImage)
        # pylab.draw()
        # pylab.show()
        # stop

        print>>log, "Splitting model image"
        self.FacetMachine.ImToFacets(self.ModelImage)
        
        NFacets=len(self.FacetMachine.DicoImager)

        DelFacet=np.zeros((NFacets,),bool)
        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            ModelFacet=self.FacetMachine.DicoImager[iFacet]["ModelFacet"]
            if np.max(np.abs(ModelFacet))==0: DelFacet[iFacet]=1
        D={}
        iFacetNew=0
        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            if DelFacet[iFacet]==False:
                D[iFacetNew]=self.FacetMachine.DicoImager[iFacet]
                iFacetNew+=1
            else:
                print>>log, "Facet [%i] is empty, removing it from direction list"%iFacet
        self.FacetMachine.DicoImager=D

        NFacets=len(self.FacetMachine.DicoImager)
        self.NDirs=NFacets
        self.Dirs=range(self.NDirs)
        ClusterCat=np.zeros((len(self.Dirs),),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)
        self.DicoImager=self.FacetMachine.DicoImager
        ClusterCat.Cluster=np.arange(NFacets)
        self.ClusterCat=ClusterCat

        #self.BuildGridsSerial()
        self.BuildGridsParallel()

        self.SM.NDir=self.NDirs
        self.SM.Dirs=self.Dirs
        self.SM.ClusterCat=self.ClusterCat
        self.SM.GD=self.FacetMachine.GD
        self.SM.DicoImager=self.FacetMachine.DicoImager
        self.SM.GD["Compression"]["CompDeGridMode"]=0
        self.SM.rac=self.VS.MS.rac
        self.SM.decc=self.VS.MS.decc

        del(self.ModelImage)
        #del(self.VS,self.FacetMachine)

    def BuildGridsSerial(self):
        print>>log, "Building the grids"
        ClusterCat=self.ClusterCat
        ListGrid=[]
        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            GM=self.FacetMachine.GiveGM(iFacet)
            ModelFacet=self.FacetMachine.DicoImager[iFacet]["ModelFacet"]
            ClusterCat.SumI[iFacet]=np.sum(ModelFacet)
            Grid=GM.dtype(GM.setModelIm(ModelFacet))
            ra,dec=self.FacetMachine.DicoImager[iFacet]["RaDec"]
            ClusterCat.ra[iFacet]=ra
            ClusterCat.dec[iFacet]=dec
            del(self.FacetMachine.DicoImager[iFacet]["ModelFacet"])
            ListGrid.append(Grid)

        
        NpShared.PackListArray("%sGrids"%(self.IdSharedMem),ListGrid)
        del(self.ModelImage)
        #del(self.VS,self.FacetMachine)


    def BuildGridsParallel(self):
        print>>log, "Building the grids"
        ClusterCat=self.ClusterCat
        ListGrid=[]

        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            ModelFacet=self.FacetMachine.DicoImager[iFacet]["ModelFacet"]
            _=NpShared.ToShared("%sModelFacet.%3.3i"%(self.IdSharedMem,iFacet),ModelFacet)


        NCPU=self.GD["Parallel"]["NCPU"]

        NFacets=len(self.DicoImager.keys())
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        NJobs=NFacets
        for iFacet in range(NFacets):
            work_queue.put(iFacet)

        GM=self.FacetMachine.GiveGM(0)
        argsImToGrid=(GM.GridShape,GM.PaddingInnerCoord,GM.OverS,GM.Padding,GM.dtype)


        workerlist=[]
        for ii in range(NCPU):
            W=Worker(work_queue, result_queue,argsImToGrid=argsImToGrid,
                     IdSharedMem=self.IdSharedMem)
            workerlist.append(W)
            workerlist[ii].start()


        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Make Grids ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NFacets))
        iResult=0

        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))


        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

            
        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            ClusterCat.SumI[iFacet]=np.sum(ModelFacet)
            Grid=NpShared.GiveArray("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet))
            ra,dec=self.FacetMachine.DicoImager[iFacet]["RaDec"]
            ClusterCat.ra[iFacet]=ra
            ClusterCat.dec[iFacet]=dec
            del(self.FacetMachine.DicoImager[iFacet]["ModelFacet"])
            ListGrid.append(Grid)
            

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
                 IdSharedMem=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.IdSharedMem=IdSharedMem
        self.SharedMemNameSphe="%sSpheroidal"%(self.IdSharedMem)
        self.ifzfCF=NpShared.GiveArray(self.SharedMemNameSphe)
        self.ClassImToGrid=ClassImToGrid(*argsImToGrid,ifzfCF=self.ifzfCF)

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            try:
                iFacet = self.work_queue.get()
            except:
                break


            ModelFacet=NpShared.GiveArray("%sModelFacet.%3.3i"%(self.IdSharedMem,iFacet))
            Grid=self.ClassImToGrid.setModelIm(ModelFacet)
            _=NpShared.ToShared("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet),Grid)
            self.result_queue.put({"Success":True})
