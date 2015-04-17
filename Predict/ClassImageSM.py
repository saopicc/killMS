
import numpy as np
from DDFacet.Imager.ClassDeconvMachine import ClassImagerDeconv
from pyrap.images import image
from killMS2.Array import NpShared

from killMS2.Other import MyLogger
log=MyLogger.getLogger("ClassImageSM")

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
        ListGrid=[]
        
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

        ClusterCat.Cluster=np.arange(NFacets)
        print>>log, "Building the grids"
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

        self.SM.NDir=self.NDirs
        self.SM.Dirs=self.Dirs
        self.SM.ClusterCat=ClusterCat
        self.SM.GD=self.FacetMachine.GD
        self.SM.DicoImager=self.FacetMachine.DicoImager
        self.SM.GD["Compression"]["CompDeGridMode"]=0
        self.SM.rac=self.VS.MS.rac
        self.SM.decc=self.VS.MS.decc
