#!/usr/bin/env python

import optparse
import pickle
import numpy as np
import pylab
import os
import glob
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassInterpol")
SaveName="last_InterPol.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFilesIn',help='Solution name patern. For example "*.MS/killMS.lala.npz')
    group.add_option('--SolFileOut',help='Name of the output file')
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)

class ClassMergeSols():
    def __init__(self,ListFilesSols):
        self.ListDictSols=[np.load(FSol) for FSol in sorted(ListFilesSols)]
        self.ListJonesSols=[DictSols["Sols"].view(np.recarray) for DictSols in self.ListDictSols]
        self.SortInFreq()
        self.NSolsFile=len(self.ListDictSols)
        self.CheckConformity()

    def CheckConformity(self):
        print>>log,"  Checking solution conformity"
        for iSol in range(1,self.NSolsFile):
            C0=np.allclose(self.ListJonesSols[0].t0,self.ListJonesSols[iSol].t0)
            C1=np.allclose(self.ListJonesSols[0].t1,self.ListJonesSols[iSol].t1)
            #C2=np.allclose(self.ListDictSols[0]["StationNames"],self.ListDictSols[iSol]["StationNames"])
            C3=np.allclose(self.ListDictSols[0]["BeamTimes"],self.ListDictSols[iSol]["BeamTimes"])
            #C4=np.allclose(self.ListDictSols[0]["ClusterCat"],self.ListDictSols[iSol]["ClusterCat"])
            if not (C0 and C1 and C3):
                raise RuntimeError("Only merging in frequency")

    def SortInFreq(self):
        print>>log,"  Sorting in frequency"
        ListCentralFreqs=[np.mean(DicoSols["FreqDomains"]) for DicoSols in self.ListDictSols]
        indSort=np.argsort(ListCentralFreqs)
        self.ListDictSols=[self.ListDictSols[iSort] for iSort in indSort]
        self.ListJonesSols=[self.ListJonesSols[iSort] for iSort in indSort]

    def NormMatrices(self,G):
        print>>log,"  Normalising Jones matrices (to antenna 0)"
        nt,nch,na,nd,_,_=G.shape
        for ich in range(nch):
            #print "%i,%i"%(ich,nch)
            for it in range(nt):
                for iDir in range(nd):
                    Gt=G[it,ich,:,iDir,:,:]
                    u,s,v=np.linalg.svd(Gt[0])
                    U=np.dot(u,v)
                    for iAnt in range(0,na):
                        Gt[iAnt,:,:]=np.dot(U.T.conj(),Gt[iAnt,:,:])
        return G

    def SaveDicoMerged(self,FileOut):

        print>>log,"  Merging solutions in frequency"
        NTimes=self.ListJonesSols[0].t0.size
        ListNFreqs=[DictSols["FreqDomains"].shape[0] for DictSols in self.ListDictSols]
        NFreqsOut=sum(ListNFreqs)

        #'ModelName', 'StationNames', 'BeamTimes', 'SourceCatSub', 'ClusterCat', 'Sols', 'SkyModel', 'FreqDomains', 

        Sols0=self.ListJonesSols[0]
        Dico0=self.ListDictSols[0]

        DicoOut={}
        DicoOut['ModelName']=Dico0['ModelName']
        DicoOut['StationNames']=Dico0['StationNames']
        DicoOut['BeamTimes']=Dico0['BeamTimes']
        DicoOut['SourceCatSub']=Dico0['SourceCatSub']
        DicoOut['ClusterCat']=Dico0['ClusterCat']
        DicoOut['SkyModel']=Dico0['SkyModel']

        nt,_,na,nd,_,_=Sols0.G.shape
        SolsOut=np.zeros((nt,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(NFreqsOut,na,nd,2,2)),("Stats",np.float32,(NFreqsOut,na,4))])
        SolsOut=SolsOut.view(np.recarray)
        SolsOut.t0=Sols0.t0
        SolsOut.t1=Sols0.t1
        iFreq=0
        FreqDomains=np.zeros((NFreqsOut,2),np.float64)
        for iSol in range(self.NSolsFile):
            ThisG=self.ListJonesSols[iSol].G
            ThisNFreq=ThisG.shape[1]
            SolsOut.G[:,iFreq:iFreq+ThisNFreq,:,:,:,:]=ThisG
            FreqDomains[iFreq:iFreq+ThisNFreq,:]=self.ListDictSols[iSol]["FreqDomains"]
            print>>log, "Freq Channels: %s"%str(np.mean(self.ListDictSols[iSol]["FreqDomains"],axis=1).ravel().tolist())
            iFreq+=ThisNFreq

        self.NormMatrices(SolsOut.G)

        DicoOut['FreqDomains']=FreqDomains
        DicoOut['Sols']=SolsOut

        if not ".npz" in FileOut: FileOut+=".npz"
        print>>log,"  Saving interpolated solutions in: %s"%FileOut
        np.savez(FileOut,**DicoOut)


def test():
    ll=glob.glob("/media/6B5E-87D0/killMS_Pack/DDFacet_Master/TestGA/TestSplitMerge/000?.MS/killMS.KAFCA.sols.npz")
    CM=ClassMergeSols(ll)
    DicoOut=CM.SaveDicoMerged("MergedSols")
    stop

def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)
    #FileName="killMS.KAFCA.sols.npz"


    SolsFiles=options.SolsFilesIn
    
    if ".txt" in SolsFiles:
        f=open(SolsFiles)
        Ls=f.readlines()
        f.close()
        MSName=[]
        for l in Ls:
            ll=l.replace("\n","")
            MSName.append(ll)
        lMS=MSName
        ListSolsFile=lMS
    elif "*" in SolsFiles:
        ListSolsFile=sorted(glob.glob(SolsFiles))

    if options.SolFileOut is None:
        raise RuntimeError("You have to specify In/Out solution file names")
    print>>log, "Running Merge on the following SolsFile:"
    for SolsName in ListSolsFile:
        print>>log, "  %s"%SolsName
    CM=ClassMergeSols(ListSolsFile)
    DicoOut=CM.SaveDicoMerged(options.SolFileOut)


if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)
    main(options=options)

