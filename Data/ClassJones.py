import numpy as np
from killMS2.Other import MyLogger
log=MyLogger.getLogger("ClassJones")
from killMS2.Other import ModColor

class ClassJones():
    def __init__(self,GD):
        self.GD=GD

    def ReClusterSkyModel(self,SM):

        SolRefFile=self.GD["PreApply"]["PreApplySols"][0]
        print>>log, ModColor.Str("Re-clustering input SkyModel to match %s clustering"%SolRefFile)
        
        ClusterCat0=np.load(SolRefFile)["ClusterCat"]
        ClusterCat0=ClusterCat0.view(np.recarray)

        lc=ClusterCat0.l
        mc=ClusterCat0.m
        lc=lc.reshape((1,lc.size))
        mc=mc.reshape((1,mc.size))
       

        l=SM.SourceCat.l
        m=SM.SourceCat.m
        l=l.reshape((l.size,1))
        m=m.reshape((m.size,1))
        d=np.sqrt((l-lc)**2+(m-mc)**2)
        Cluster=np.argmin(d,axis=1)
        print SM.SourceCat.Cluster
        SM.SourceCat.Cluster[:]=Cluster[:]
        SM.ClusterCat=ClusterCat0
        print SM.SourceCat.Cluster

        NDir=lc.size
        for iDir in range(NDir):
            ind=(SM.SourceCat.Cluster==iDir)
            SM.ClusterCat.SumI[iDir]=np.sum(SM.SourceCat.I[ind])

