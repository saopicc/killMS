#!/usr/bin/env python
import optparse
import pickle

import numpy as np
import pylab
from pyrap.tables import table
import Array.ModLinAlg
from DDFacet.Other import logger
log = logger.getLogger("AQWeight")
from killMS.Data import ClassMS
from DDFacet.Array import shared_dict
APP=None
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
import Weights.W_DiagBL
import Weights.W_AntFull
import Weights.W_Jones
import Weights.W_Imag

def read_options():
    desc="""Run MCMC """
    
    opt = optparse.OptionParser(usage='Usage: %prog <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--ListMSName',type=str,help='',default="")
    group.add_option('--DataCol',type=str,help='',default="DATA_DI_CORRECTED")
    group.add_option('--PredictCol',type=str,help='',default="PREDICT")
    group.add_option('--WeightCol',type=str,help='',default="IMAGING_WEIGHT2")
    group.add_option('--SolsFile',type=str,help='',default=None)
    group.add_option('--CovType',type=str,help='',default="DiagBL")
    group.add_option('--TBinBox',type=int,help='',default=20)
    group.add_option('--ds9Regions',type=str,help='',default="")
    
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options


class AQW():
    def __init__(self,**kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        self.iCurrentMS=0

        if ".txt" in self.ListMSName:
            self.ListMSName = [ l.strip() for l in open(self.ListMSName).readlines() ]
        else:
            self.ListMSName=[self.ListMSName]

        self.MS0=ClassMS.ClassMS(self.ListMSName[0],Col=self.DataCol,DoReadData=False,ReadUVWDT=False)

        print>>log, "Running reweighting on the following MSNames:"
        for MSName in self.ListMSName:
            print>>log, "  %s"%MSName

        self.Normalise=False
        self.DictName="DATA"
        self.DoNeedVisibilities=True
        if self.CovType=="DiagBL":
            self.CovMachine=Weights.W_DiagBL.ClassCovMat(**kwargs)
            self.Normalise=True
        elif self.CovType=="AntFull":
            self.CovMachine=Weights.W_AntFull.ClassCovMat(**kwargs)
        elif self.CovType=="Jones":
            self.CovMachine=Weights.W_Jones.ClassCovMat(**kwargs)
            self.DoNeedVisibilities=False
        elif self.CovType=="VarImag":
            self.CovMachine=Weights.W_Imag.ClassCovMat(**kwargs)
            self.CovMachine.setMS(self.MS0)
            self.CovMachine.setDirs()

        APP.registerJobHandlers(self.CovMachine)
        APP.startWorkers()
       


    def LoadNextMS(self):
        if self.iCurrentMS==len(self.ListMSName):
            print>>log,"Reached end of MSList"
            return False
        MSName=self.ListMSName[self.iCurrentMS]
        self.MS=ClassMS.ClassMS(MSName,Col=self.DataCol,DoReadData=False,ReadUVWDT=False)
        
        t=table(MSName,ack=False)
        d=None
        f=None
        
        if self.DoNeedVisibilities:
            print>>log,"Reading data column %s"%self.DataCol
            d=t.getcol(self.DataCol)
            
            print>>log,"Reading model column %s"%self.PredictCol
            p=t.getcol(self.PredictCol)
            print>>log,"Reading flags"
            f=t.getcol("FLAG")
            f[:,:,1]=1
            f[:,:,2]=1
            nr,nch,_=f.shape
        
            d[np.isnan(d)]=0.
            do=d.copy()
            d-=p
            p[(d==0)|(p==0)]=1.
            p/=np.abs(p)
            d/=p
            d[f]=0.
            do[f]=0.

        self.DATA = shared_dict.create("DATA")

        self.DATA["data"]=d
        self.DATA["radec"]=self.MS.radec
        self.DATA["data_orig"]=do
        self.DATA["flag"]=f
        self.DATA["freqs"]=self.MS.ChanFreq.ravel()
        self.DATA["uvw"]=t.getcol("UVW")
        
        print>>log,"Reading other stuff"
        self.DATA["A0"]=t.getcol("ANTENNA1")
        self.DATA["times"]=t.getcol("TIME")
        self.DATA["times_unique"]=np.sort(np.unique(self.DATA["times"]))
        self.DATA["A1"]=t.getcol("ANTENNA2")
        self.DATA["W"]=t.getcol("IMAGING_WEIGHT")#np.zeros_like(t.getcol("IMAGING_WEIGHT"))
        #self.DATA["N"]=np.zeros_like(self.DATA["W"])
        self.na=self.DATA["na"]=np.max(self.DATA["A0"])+1
        print>>log,"There are %i antennas"%self.na
        t.close()

        print>>log, "Add weight column %s"%self.WeightCol
        self.MS.AddCol(self.WeightCol,ColDesc="IMAGING_WEIGHT")
        return True



    def killWorkers(self):
        print>>log, ModColor.Str("Killing workers")
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()

    def calcWeights(self):
        while True:
            if not self.LoadNextMS(): break
            self.CovMachine.giveWeigthParallel()
            #self.CovMachine.giveWeigth()
            self.CovMachine.Finalise()
            self.Write()
            self.iCurrentMS+=1
        self.killWorkers()
        



    def Write(self):

        
        print>>log,"Writting weights in column %s"%self.WeightCol
        t=table(self.ListMSName[self.iCurrentMS],ack=False,readonly=False)
        t.putcol(self.WeightCol,self.DATA["W"])
        t.close()


            
#########################################

def main(options=None):
    if options is None:
        f = open("last_param.obj",'rb')
        options = pickle.load(f)
    

    

    MM=AQW(**options.__dict__)
    MM.calcWeights()

    

if __name__=="__main__":
    OP=read_options()
    main(OP)