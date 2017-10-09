import numpy as np
from killMS.Other import MyLogger
log=MyLogger.getLogger("ClassSolPredictMachine")
from killMS.Other import ModColor

def D(a0,a1,b0=None,b1=None):
    if b0 is not None:
        d=np.sqrt((a0.reshape((-1,1))-a1.reshape((1,-1)))**2+(b0.reshape((-1,1))-b1.reshape((1,-1)))**2)
    else:
        d=np.sqrt((a0.reshape((-1,1))-a1.reshape((1,-1)))**2)
    return d

        
class ClassSolPredictMachine():
    def __init__(self,GD):
        self.GD=GD
        FileName=self.GD["KAFCA"]["EvolutionSolFile"]
        print>>log, "Reading solution file %s"%FileName
        self.DicoSols=np.load(FileName)
        self.Sols=self.DicoSols["Sols"]
        self.Sols=self.Sols.view(np.recarray)
        self.FreqDomains=self.DicoSols["FreqDomains"]
        self.MeanFreqDomains=np.mean(self.FreqDomains,axis=1)
        self.t0=self.Sols.t0
        self.t1=self.Sols.t1
        self.tmean=(self.t0+self.t1)/2.
        self.G=self.Sols.G
        self.ClusterCat=Sols["ClusterCat"]
        self.ClusterCat=self.ClusterCat.view(np.recarray)
        self.raSol=self.ClusterCat.ra
        self.decSol=self.ClusterCat.dec

        # nt,nch,na,nd,_,_=self.G.shape
        

    def GiveClosestSol(self,t,freq_domains,ras,decs):
        ns=np_ra.size
        d=D(ras,self.raSol,decs,self.decSol)
        ind_d=np.argmin(d,axis=1)
        
        mean_freqs=np.mean(freq_domains,axis=1)
        dfreq=D(mean_freqs,self.MeanFreqDomains)
        ind_freq=np.argmin(d,axis=1)
        
        dtime=D(np.array([t]),self.MeanFreqDomains)
        ind_time=np.argmin(dtime,axis=1)
        
        # nChan,na,nd,npolx,npoly
        G=self.G[ind_time,ind_freq,:,ind_d,:,:]

        return G[0]
        
