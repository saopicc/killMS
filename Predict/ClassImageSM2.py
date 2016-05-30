
import numpy as np
from DDFacet.Imager.ClassDeconvMachine import ClassImagerDeconv
from pyrap.images import image
from killMS2.Array import NpShared
from killMS2.Other import reformat

from killMS2.Other import MyLogger
log=MyLogger.getLogger("ClassImageSM")
from killMS2.Other.progressbar import ProgressBar
from DDFacet.ToolsDir.GiveEdges import GiveEdges
#from DDFacet.Imager.ClassModelMachine import ClassModelMachine
#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#from DDFacet.Imager.ClassModelMachineGA import ClassModelMachine
from DDFacet.Imager.ModModelMachine import GiveModelMachine

from DDFacet.Imager.ClassImToGrid import ClassImToGrid
import DDFacet.Other.MyPickle
import os
from DDFacet.Data import ClassVisServer

class ClassImageSM():
    def __init__(self):
        self.Type="Image"


class ClassPreparePredict(ClassImagerDeconv):

    def __init__(self,BaseImageName,VS,*args,**kwargs):
        ClassImagerDeconv.__init__(self,**kwargs)

        self.BaseImageName=BaseImageName
        self.FileDicoModel="%s.DicoModel"%self.BaseImageName
        self.ModelImageName="%s.model.fits"%self.BaseImageName

        self.VS=VS
        # DC=self.GD
        # MSName=DC["VisData"]["MSName"]
        # self.VS=ClassVisServer.ClassVisServer(MSName,
        #                                       ColName=DC["VisData"]["ColName"],
        #                                       TVisSizeMin=DC["VisData"]["TChunkSize"]*60,
        #                                       #DicoSelectOptions=DicoSelectOptions,
        #                                       TChunkSize=DC["VisData"]["TChunkSize"],
        #                                       IdSharedMem=self.IdSharedMem,
        #                                       Robust=DC["ImagerGlobal"]["Robust"],
        #                                       Weighting=DC["ImagerGlobal"]["Weighting"],
        #                                       Super=DC["ImagerGlobal"]["Super"],
        #                                       DicoSelectOptions=dict(DC["DataSelection"]),
        #                                       NCPU=self.GD["Parallel"]["NCPU"],
        #                                       GD=self.GD)

        #self.GD["GAClean"]["GASolvePars"]=["S","Alpha"]
        
        self.IdSharedMem=kwargs["IdSharedMem"]
        self.SM=ClassImageSM()

        if self.GD["GDkMS"]["ImageSkyModel"]["NodesFile"]!=None:
            self.GD["CatNodes"]=self.GD["GDkMS"]["ImageSkyModel"]["NodesFile"]
            self.GD["DDESolutions"]["DDSols"]=""
            
        self.InitFacetMachine()
        self.LoadModel()

    def LoadModel(self):

        
        ClassModelMachine,DicoModel=GiveModelMachine(self.FileDicoModel)
        try:
            self.GD["GAClean"]["GASolvePars"]=DicoModel["SolveParam"]
        except:
            self.GD["GAClean"]["GASolvePars"]=["S","Alpha"]
            DicoModel["SolveParam"]=self.GD["GAClean"]["GASolvePars"]
        self.MM=ClassModelMachine(self.GD)
        self.MM.FromDico(DicoModel)
        #ModelImage0=self.MM.GiveModelImage(np.mean(self.VS.MS.ChanFreq))
        #self.MM.CleanNegComponants(box=15,sig=1)
        if self.GD["GDkMS"]["ImageSkyModel"]["MaskImage"]!=None:
            self.MM.CleanMaskedComponants(self.GD["GDkMS"]["ImageSkyModel"]["MaskImage"])
        self.ModelImage=self.MM.GiveModelImage(np.mean(self.VS.MS.ChanFreq))

        # print "im!!!!!!!!!!!!!!!!!!!!!!!"
        # im=image("ModelImage.fits")
        # data=im.getdata()
        # nch,npol,nx,_=data.shape
        # for ch in range(nch):
        #     for pol in range(npol):
        #         data[ch,pol]=data[ch,pol].T[::-1]
        # self.ModelImage=data

        self.FacetMachine.ToCasaImage(self.ModelImage,ImageName="Model_kMS",Fits=True)
        #stop
        self.ModelImage=NpShared.ToShared("%sModelImage"%(self.IdSharedMem),self.ModelImage)
        #del(data)
        self.DicoImager=self.FacetMachine.DicoImager
        
        
        NFacets=len(self.FacetMachine.DicoImager)
        self.NFacets=NFacets
        #self.NDirs=NFacets
        #self.Dirs=range(self.NDirs)

        # SolsFile=self.GD["DDESolutions"]["DDSols"]
        # if not(".npz" in SolsFile):
        #     ThisMSName=reformat.reformat(os.path.abspath(self.VS.MSName),LastSlash=False)
        #     SolsFile="%s/killMS.%s.sols.npz"%(self.VS.MSName,SolsFile)
        #     #SolsFile="BOOTES24_SB100-109.2ch8s.ms/killMS.KAFCA.Scalar.50Dir.0.1P.BriggsSq.PreCuster4.sols.npz"

        # DicoSolsFile=np.load(SolsFile)
        # ClusterCat=DicoSolsFile["ClusterCat"]
        # ClusterCat=ClusterCat.view(np.recarray)
        
        
        #DicoFacetName="%s.DicoFacet"%self.BaseImageName
        #DicoFacet=DDFacet.Other.MyPickle.Load(DicoFacetName)
        
        NodeFile="%s.NodesCat.npy"%self.BaseImageName
        NodesCat=np.load(NodeFile)
        NodesCat=NodesCat.view(np.recarray)

        self.NDir=NodesCat.shape[0]

        ClusterCat=np.zeros((self.NDir,),dtype=[('Name','|S200'),
                                                        ('ra',np.float),('dec',np.float),
                                                        ('l',np.float),('m',np.float),
                                                        ('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)
        ClusterCat.l=NodesCat.l
        ClusterCat.m=NodesCat.m
        ClusterCat.ra=NodesCat.ra
        ClusterCat.dec=NodesCat.dec

        self.DicoImager=self.FacetMachine.DicoImager
        self.ClusterCat=ClusterCat
        self.ClusterCat.SumI=0.


        #ind=np.where(self.ClusterCat.SumI!=0)[0]
        #self.ClusterCat=self.ClusterCat[ind].copy()
        #NFacets=self.ClusterCat.shape[0]
        #print>>log, "  There are %i non-zero facets"%NFacets

        NFacets=len(self.FacetMachine.DicoImager)
        lFacet=np.zeros((NFacets,),np.float32)
        mFacet=np.zeros_like(lFacet)
        for iFacet in range(NFacets):
            l,m=self.FacetMachine.DicoImager[iFacet]["lmShift"]
            lFacet[iFacet]=l
            mFacet[iFacet]=m

        NDir=ClusterCat.l.size
        d=np.sqrt((ClusterCat.l.reshape((NDir,1))-lFacet.reshape((1,NFacets)))**2+
                  (ClusterCat.m.reshape((NDir,1))-mFacet.reshape((1,NFacets)))**2)
        idDir=np.argmin(d,axis=0)
        for iFacet in range(NFacets):
            self.FacetMachine.DicoImager[iFacet]["iDirJones"]=idDir[iFacet]


        

        for iFacet in range(NFacets):
            self.FacetMachine.SpacialWeigth[iFacet]=NpShared.ToShared("%sSpacialWeight_%3.3i"%(self.IdSharedMem,iFacet),self.FacetMachine.SpacialWeigth[iFacet])

        print>>log, "  Splitting model image"
        self.BuildGridsParallel()


        #self.BuildGridsSerial()
        #self.BuildGridsParallel()

        NFacets=self.ClusterCat.shape[0]
        self.SM.NDir=self.NDirs
        self.SM.Dirs=self.Dirs
        print>>log, "  There are %i non-zero directions"%self.SM.NDir
        self.SM.ClusterCat=self.ClusterCat
        self.SM.DicoJonesDirToFacet=self.DicoJonesDirToFacet
        self.SM.GD=self.FacetMachine.GD
        self.SM.DicoImager=self.FacetMachine.DicoImager
        self.SM.GD["Compression"]["CompDeGridMode"]=0
        self.SM.rac=self.VS.MS.rac
        self.SM.decc=self.VS.MS.decc
        
        # self.SM.ChanMappingDegrid=self.VS.DicoMSChanMappingDegridding[0]





    def BuildGridsParallel(self):
        print>>log, "  Building the grids"
        ListGrid=[]



        NCPU=self.GD["Parallel"]["NCPU"]

        NFacets=len(self.DicoImager.keys())
        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        NJobs=NFacets
        for iFacet in range(NFacets):
            work_queue.put(iFacet)


        self.FacetMachine.BuildFacetNormImage()#GiveNormImage()
        _=NpShared.ToShared("%sNormImage"%self.IdSharedMem,self.FacetMachine.NormImage)

        GM=self.FacetMachine.GiveGM(0)
        argsImToGrid=(GM.GridShape,GM.PaddingInnerCoord,GM.OverS,GM.Padding,GM.dtype)
        
        workerlist=[]
        # FacetMode="Sharp"
        FacetMode="Fader"
        

        # #####################################"
        # #### test
        # self.SharedMemNameSphe="%sSpheroidal"%(self.IdSharedMem)
        # self.ifzfCF=NpShared.GiveArray(self.SharedMemNameSphe)
        # self.ClassImToGrid=ClassImToGrid(*argsImToGrid,ifzfCF=self.ifzfCF)
        
        # Image=NpShared.GiveArray("%sModelImage"%(self.IdSharedMem))

        # _,_,nx,ny=Image.shape
        # x,y=np.mgrid[0:nx,0:ny]
        # x=x.reshape((1,1,nx,ny))
        # y=y.reshape((1,1,nx,ny))
        
        # #Image[:,:,:,:]=np.random.randn(*(Image.shape))
        # #Image=np.sqrt(x**2+2.*y**2)

        # for iFacet in range(9):
        #     # Grid0,SumFlux0=self.ClassImToGrid.GiveGridSharp(Image,self.DicoImager,iFacet)
            
        #     # NormImage=NpShared.GiveArray("%sNormImage"%self.IdSharedMem)
        #     # Grid1,SumFlux1=self.ClassImToGrid.GiveGridFader(Image,self.DicoImager,iFacet,NormImage)

        #     # Image=NpShared.GiveArray("%sModelImage"%(self.IdSharedMem))

        #     #Grid,SumFlux=self.ClassImToGrid.GiveGridFader(Image,self.DicoImager,iFacet,NormImage)
        #     SharedMemName="%sSpheroidal.Facet_%3.3i"%(self.IdSharedMem,iFacet)
        #     SPhe=NpShared.GiveArray(SharedMemName)
        #     SpacialWeight=NpShared.GiveArray("%sSpacialWeight_%3.3i"%(self.IdSharedMem,iFacet))
        #     NormImage=NpShared.GiveArray("%sNormImage"%self.IdSharedMem)

        #     Im2Grid=ClassImToGrid(OverS=self.GD["ImagerCF"]["OverS"],GD=self.GD)
        #     #Grid,SumFlux=Im2Grid.GiveModelTessel(Image,self.DicoImager,iFacet,NormImage,SPhe,SpacialWeight,ToGrid=True)

        #     ok,ModelCutOrig,ModelCutOrig_GNorm,ModelCutOrig_SW,ModelCutOrig_Sphe,ModelCutOrig_GNorm_SW_Sphe_CorrT=Im2Grid.GiveModelTessel(Image,self.DicoImager,iFacet,NormImage,SPhe,SpacialWeight,ToGrid=True)
        #     if ok==False: continue

        #     print "================= %i"%iFacet
        #     #print np.where(Grid0==np.max(Grid0)),np.max(Grid0)
        #     #print np.where(Grid1==np.max(Grid1)),np.max(Grid1)
        #     #print np.max(np.abs(Grid0-Grid1))

        #     #vmin,vmax=0,13
        #     #vmin=0
        #     #vmax=2000

        #     x,y=np.where(ModelCutOrig==np.max(ModelCutOrig))

        #     import pylab
        #     pylab.clf()
        #     ax=pylab.subplot(2,2,1)
        #     pylab.imshow(ModelCutOrig.real,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        #     pylab.scatter(y,x)
        #     pylab.title("max=%f"%np.max(ModelCutOrig_GNorm_SW_Sphe_CorrT))
        #     #pylab.colorbar()
        #     pylab.subplot(2,2,2,sharex=ax,sharey=ax)
        #     pylab.imshow(ModelCutOrig_GNorm,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        #     pylab.colorbar()
        #     pylab.subplot(2,2,3,sharex=ax,sharey=ax)
        #     pylab.imshow(ModelCutOrig_SW,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        #     pylab.colorbar()
        #     pylab.subplot(2,2,4,sharex=ax,sharey=ax)
        #     pylab.imshow(ModelCutOrig_Sphe,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        #     pylab.colorbar()

        #     pylab.draw()
        #     pylab.show()

        #     #print Grid1[0,0,nx/2,ny/2].real-Grid0[0,0,nx/2,ny/2].real
        # stop

        # ####



        for ii in range(NCPU):
            W=Worker(work_queue, result_queue,argsImToGrid=argsImToGrid,
                     IdSharedMem=self.IdSharedMem,
                     DicoImager=self.FacetMachine.DicoImager,FacetMode=FacetMode,
                     GD=self.GD)#"Fader")
            workerlist.append(W)
            workerlist[ii].start()


        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Make Grids ", HeaderSize=10,TitleSize=13)
        pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,NFacets))
        iResult=0

        ClusterCat=self.ClusterCat
        DicoJonesDirToFacet={}
        for iDir in range(self.ClusterCat.shape[0]):
            DicoJonesDirToFacet[iDir]={}
            DicoJonesDirToFacet[iDir]["FacetsIDs"]=[]
            DicoJonesDirToFacet[iDir]["SumFlux"]=0.

        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
                iFacet=DicoResult["iFacet"]
                iDirJones=self.FacetMachine.DicoImager[iFacet]["iDirJones"]
                ClusterCat.SumI[iDirJones]+=np.real(DicoResult["SumFlux"])
                self.FacetMachine.DicoImager[iFacet]["SumFlux"]=np.real(DicoResult["SumFlux"])
                if self.FacetMachine.DicoImager[iFacet]["SumFlux"]!=0.:
                    DicoJonesDirToFacet[iDirJones]["FacetsIDs"].append(iFacet)
                    DicoJonesDirToFacet[iDirJones]["SumFlux"]+=np.real(DicoResult["SumFlux"])
                    
                #ra,dec=self.FacetMachine.DicoImager[iFacet]["RaDec"]
                #l0,m0=self.FacetMachine.DicoImager[iFacet]["l0m0"]
                #l,m=self.FacetMachine.DicoImager[iFacet]["lmShift"]
                #ClusterCat.l[iFacet]=l
                #ClusterCat.m[iFacet]=m

                #ClusterCat.ra[iFacet]=ra
                #ClusterCat.dec[iFacet]=dec

            NDone=iResult
            intPercent=int(100*  NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))


        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()




        for iFacet in sorted(self.FacetMachine.DicoImager.keys()):
            Grid=NpShared.GiveArray("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet))
            ListGrid.append(Grid)

        self.DicoImager=self.FacetMachine.DicoImager
        #self.ClusterCat=ClusterCat[DelFacet==0].copy()
        #NFacets=self.ClusterCat.shape[0]
        #self.ClusterCat.Cluster=np.arange(NFacets)
        self.DicoJonesDirToFacet=DicoJonesDirToFacet
        
        
        D={}
        iDirNew=0

        self.ClusterCatOrig=self.ClusterCat.copy()
        self.NDirsOrig=self.ClusterCat.shape[0]

        self.NDirs=self.ClusterCat.shape[0]
        Keep=np.zeros((self.NDirs,),bool)
        for iDirJones in sorted(DicoJonesDirToFacet.keys()):
            if self.DicoJonesDirToFacet[iDirJones]["SumFlux"]==0:
                print>>log,"  Remove Jones direction %i"%(iDirJones)
            else:
                D[iDirNew]=self.DicoJonesDirToFacet[iDirJones]
                iDirNew+=1
                Keep[iDirJones]=1
        self.MapClusterCatOrigToCut=Keep

        self.DicoJonesDirToFacet=D
        self.ClusterCat=self.ClusterCat[Keep].copy()
        #self.Dirs=self.ClusterCat.Cluster.tolist()
        #self.Dirs=[iDir for iDir in DicoJonesDirToFacet.keys() if DicoJonesDirToFacet[iDir]["SumFlux"]!=0.]
        #self.Dirs=[iDir for iDir in DicoJonesDirToFacet.keys() if DicoJonesDirToFacet[iDir]["SumFlux"]!=0.]

        self.Dirs=self.DicoJonesDirToFacet.keys()
        self.NDirs=len(self.Dirs)
        
        # print self.ClusterCat.ra
        # print self.DicoJonesDirToFacet
        
        # from DDFacet.Other import MyPickle
        # np.save("ClusterCat",self.ClusterCat)
        # MyPickle.Save(self.DicoJonesDirToFacet,"DicoJonesDirToFacet")
        # MyPickle.Save(self.FacetMachine.DicoImager,"DicoImager")

        # stop
        NpShared.PackListArray("%sGrids"%(self.IdSharedMem),ListGrid)
        NpShared.DelAll("%sModelFacet"%self.IdSharedMem)
        NpShared.DelAll("%sModelGrid"%self.IdSharedMem)
        return True


        



 
import multiprocessing
from killMS2.Predict.PredictGaussPoints_NumExpr5 import ClassPredict
class Worker(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 argsImToGrid=None,
                 IdSharedMem=None,
                 DicoImager=None,
                 FacetMode="Fader",GD=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.DicoImager=DicoImager
        self.FacetMode=FacetMode
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
            
            
            #Grid,SumFlux=self.ClassImToGrid.GiveGridFader(Image,self.DicoImager,iFacet,NormImage)
            SharedMemName="%sSpheroidal.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            SPhe=NpShared.GiveArray(SharedMemName)
            SpacialWeight=NpShared.GiveArray("%sSpacialWeight_%3.3i"%(self.IdSharedMem,iFacet))
            NormImage=NpShared.GiveArray("%sNormImage"%self.IdSharedMem)
            
            Im2Grid=ClassImToGrid(OverS=self.GD["ImagerCF"]["OverS"],GD=self.GD)
            Grid,SumFlux=Im2Grid.GiveModelTessel(Image,self.DicoImager,iFacet,NormImage,SPhe,SpacialWeight,ToGrid=True)

            #ModelSharedMemName="%sModelImage.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            #NpShared.ToShared(ModelSharedMemName,ModelFacet)



            _=NpShared.ToShared("%sModelGrid.%3.3i"%(self.IdSharedMem,iFacet),Grid)

            self.result_queue.put({"Success":True,"iFacet":iFacet,"SumFlux":SumFlux})
            
