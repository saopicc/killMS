from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("ClassSolPredictMachine")
from killMS.Other import ModColor
import os
from killMS.Other import reformat

def D(a0,a1,b0=None,b1=None):
    if b0 is not None:
        d=np.sqrt((a0.reshape((-1,1))-a1.reshape((1,-1)))**2+(b0.reshape((-1,1))-b1.reshape((1,-1)))**2)
    else:
        d=np.sqrt((a0.reshape((-1,1))-a1.reshape((1,-1)))**2)
    return d

        
class ClassSolPredictMachine():
    def __init__(self,GD):
        self.GD=GD
        # FileName=self.GD["KAFCA"]["EvolutionSolFile"]
        # if not ".npz" in FileName:
        #     FileName="%s/killMS.%s.sols.npz"%(self.GD["VisData"]["MSName"],FileName)


        SolsDir=GD["Solutions"]["SolsDir"]
        SolsName=self.GD["KAFCA"]["EvolutionSolFile"]
        MSName=os.path.abspath(self.GD["VisData"]["MSName"])
        if SolsDir is None:
            FileName="%skillMS.%s.sols.npz"%(reformat.reformat(options.MSName),SolsName)
        else:
            _MSName=reformat.reformat(MSName).split("/")[-2]
            DirName=os.path.abspath("%s%s"%(reformat.reformat(SolsDir),_MSName))
            if not os.path.isdir(DirName):
                os.makedirs(DirName)
            FileName="%s/killMS.%s.sols.npz"%(DirName,SolsName)




        log.print( "Reading solution file %s"%FileName)
        self.DicoSols=np.load(FileName)
        self.Sols=self.DicoSols["Sols"]
        self.Sols=self.Sols.view(np.recarray)
        self.FreqDomains=self.DicoSols["FreqDomains"]
        self.MeanFreqDomains=np.mean(self.FreqDomains,axis=1)
        self.t0=self.Sols.t0
        self.t1=self.Sols.t1
        self.tmean=(self.t0+self.t1)/2.
        self.G=self.Sols.G
        self.ClusterCat=self.DicoSols["ClusterCat"]
        self.ClusterCat=self.ClusterCat.view(np.recarray)
        self.raSol=self.ClusterCat.ra
        self.decSol=self.ClusterCat.dec

        # nt,nch,na,nd,_,_=self.G.shape
        

    def GiveClosestSol(self,t,freq_domains,ants,ras,decs):
        d=D(ras,self.raSol,decs,self.decSol)
        ind_dir=np.argmin(d,axis=1)
        
        mean_freqs=np.mean(freq_domains,axis=1)
        dfreq=D(mean_freqs,self.MeanFreqDomains)
        ind_freq=np.argmin(dfreq,axis=1)
        
        dtime=D(np.array([t]),self.tmean)
        ind_time=np.argmin(dtime,axis=1)

        # print ind_time
        # print self.G.shape
        # # nChan,na,nd,npolx,npoly

        nch=ind_freq.size
        ndir=ind_dir.size
        na=ants.size
        Gout=np.zeros((nch,na,ndir,2,2),np.complex64)
        ind_pol=np.arange(4)

        # Gout0=np.zeros((nch,na,ndir,2,2),np.complex64)
        # for ch in ind_freq.ravel():
        #     for ant in ants.ravel():
        #         for d in ind_dir.ravel():
        #             for polx in range(2):
        #                 for poly in range(2):
        #                     Gout0[ch,ant,d,polx,poly]=self.G[ind_time[0]][ch,ant,d,polx,poly]

        ind_time=ind_time.reshape((-1,1,1,1,1))
        ind_freq=ind_freq.reshape((1,-1,1,1,1))
        ind_ant=ants.reshape((1,1,-1,1,1))
        ind_dir=ind_dir.reshape((1,1,1,-1,1))
        ind_pol=ind_pol.reshape((1,1,1,1,-1))

        nt_G,nch_G,na_G,nd_G,_,_=self.G.shape
        index=ind_time*nch_G*na_G*nd_G*2*2 + ind_freq*na_G*nd_G*2*2 + ind_ant*nd_G*2*2 + ind_dir*2*2 + ind_pol
        Gout.flat[:]=self.G.flat[index.ravel()]

        

        # # ind_freq=ind_freq.reshape((-1,1,1,1))
        # # ind_ant=ants.reshape((1,-1,1,1))
        # # ind_dir=ind_dir.reshape((1,1,-1,1))
        # # ind_pol=ind_pol.reshape((1,1,1,-1))
        # # index= ind_freq*na*ndir*2*2 + ind_ant*ndir*2*2 + ind_dir*2*2 + ind_pol
        # # Gout.flat[:]=self.G[ind_time[0]].flat[index.ravel()]
        # print np.max(Gout0-Gout)
        # print
        # print
        # print
        # #Gout=Gout0
        
        return Gout
        
