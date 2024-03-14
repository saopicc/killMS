#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import pickle
import numpy as np
#import pylab
import os
import glob
from DDFacet.Other import logger
log=logger.getLogger("ClassInterpol")
SaveName="last_InterPol.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFilesIn',help='Solution name patern. For example "*.MS/killMS.lala.npz')
    group.add_option('--SolFileOut',help='Name of the output file')
    group.add_option('--SigmaFilterOutliers',help='Filter the outliers at this Sigma? Default is %default',default=None,type="float")
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)

def GiveMAD(X):
    med=np.median(X)
    return med,1.4826*np.median(np.abs(X-med))

class ClassMergeSols():
    def __init__(self,ListFilesSols):
        self.ListDictSols=[np.load(FSol,allow_pickle=True) for FSol in sorted(ListFilesSols)]
        self.ListJonesSols=[DictSols["Sols"].view(np.recarray) for DictSols in self.ListDictSols]
        self.SortInFreq()
        self.NSolsFile=len(self.ListDictSols)
        self.CheckConformity()


    def CheckConformity(self):
        log.print("  Checking solution conformity")
        for iSol in range(1,self.NSolsFile):
            C0=np.allclose(self.ListJonesSols[0].t0,self.ListJonesSols[iSol].t0)
            C1=np.allclose(self.ListJonesSols[0].t1,self.ListJonesSols[iSol].t1)
            #C2=np.allclose(self.ListDictSols[0]["StationNames"],self.ListDictSols[iSol]["StationNames"])
            C3=np.allclose(self.ListDictSols[0]["BeamTimes"],self.ListDictSols[iSol]["BeamTimes"])
            #C4=np.allclose(self.ListDictSols[0]["ClusterCat"],self.ListDictSols[iSol]["ClusterCat"])
            if not (C0 and C1 and C3):
                raise RuntimeError("Only merging in frequency")

    def SortInFreq(self):
        log.print("  Sorting in frequency")
        ListCentralFreqs=[np.mean(DicoSols["FreqDomains"]) for DicoSols in self.ListDictSols]
        indSort=np.argsort(ListCentralFreqs)
        self.ListDictSols=[self.ListDictSols[iSort] for iSort in indSort]
        self.ListJonesSols=[self.ListJonesSols[iSort] for iSort in indSort]
        DFs=np.array([DicoSols["FreqDomains"][:,1]-DicoSols["FreqDomains"][:,0] for DicoSols in self.ListDictSols])
        f0s=np.array([DicoSols["FreqDomains"][:,0] for DicoSols in self.ListDictSols])
        f1s=np.array([DicoSols["FreqDomains"][:,1] for DicoSols in self.ListDictSols])

        if np.max(np.abs(DFs[1::]-DFs[0:-1]))>1e-5:
            raise RuntimeError("Solutions don't have the same width")
        self.df=DFs.ravel()[0]
        log.print("  Solution channel width is %f MHz for each solution file"%(self.df/1e6))
        f0=f0s.min()
        f1=f1s.max()
        NFreqsOut=(f1-f0)/self.df
        if NFreqsOut%1!=0:
            raise RuntimeError("Solutions have got to be equally spaced in frequency")
        self.NFreqsOut=int(NFreqsOut)
        self.FreqDomainsOut=np.zeros((self.NFreqsOut,2),np.float64)
        self.FreqDomainsOut[:,0]=f0+np.arange(self.NFreqsOut)*self.df
        self.FreqDomainsOut[:,1]=f0+self.df+np.arange(self.NFreqsOut)*self.df

    def NormMatrices(self,G):
        log.print("  Normalising Jones matrices (to antenna 0)")
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

    def FilterOutliers(self,Sigma):
        G=self.DicoOut['Sols']["G"]
        nt,nf,na,nd,_,_=G.shape
        log.print("Filtering out outliers using a sigma of %f"%Sigma)
        Ga=np.abs(G)
        N=np.zeros((na,nd),np.float32)
        NMask=np.zeros((na,nd),np.float32)
        for iDir in range(nd):
            mean,sig=GiveMAD(Ga[:,:,:,iDir,0,0])
            for iAnt in range(na):
                Gas=Ga[:,:,iAnt,iDir,0,0]
                indt,indf=np.where(Gas>(mean+Sigma*sig))
                if indt.size>0:
                    self.DicoOut['Sols']["G"][indt,indf,iAnt,iDir,0,0]=mean
                    self.DicoOut['Sols']["G"][indt,indf,iAnt,iDir,1,1]=mean
                    self.DicoOut['Sols']["G"][indt,indf,iAnt,iDir,0,1]=0
                    self.DicoOut['Sols']["G"][indt,indf,iAnt,iDir,1,0]=0
                N[iAnt,iDir]=Gas.size
                NMask[iAnt,iDir]=indt.size
            
            log.print("  [Dir %2i]  Filtered %5.2f%% of points"%(iDir,100.*np.sum(NMask[iDir])/float(np.sum(N[iDir]))))
        

                
    def MakeDicoMerged(self):
        
        log.print("  Merging solutions in frequency")
        NTimes=self.ListJonesSols[0].t0.size
        ListNFreqs=[DictSols["FreqDomains"].shape[0] for DictSols in self.ListDictSols]

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
        NFreqsOut=self.NFreqsOut
        nt,_,na,nd,_,_=Sols0.G.shape
        SolsOut=np.zeros((nt,),dtype=[("t0",np.float64),("t1",np.float64),("G",np.complex64,(NFreqsOut,na,nd,2,2)),("Stats",np.float32,(NFreqsOut,na,4))])
        SolsOut=SolsOut.view(np.recarray)
        SolsOut.t0=Sols0.t0
        SolsOut.t1=Sols0.t1
        log.print("Output Solution shape: %s"%(str(SolsOut.G.shape)))
        
        Mask=np.ones(SolsOut.G.shape,np.int16)
        ArrayMSNames=np.zeros((NFreqsOut,),"|S200")
        for iSol in range(self.NSolsFile):
            ThisG=self.ListJonesSols[iSol].G
            ThisNFreq=ThisG.shape[1]
            fmean_sol0=np.mean(self.ListDictSols[iSol]["FreqDomains"][0,:])
            iFreq=np.where((fmean_sol0>self.FreqDomainsOut[:,0])&(fmean_sol0<self.FreqDomainsOut[:,1]))[0]
            if iFreq.size!=1:
                raise RuntimeError("That's a bug")
            iFreq=iFreq.ravel()[0]
            SolsOut.G[:,iFreq:iFreq+ThisNFreq,:,:,:,:]=ThisG
            Mask[:,iFreq:iFreq+ThisNFreq]=0

            log.print( "Freq Channels: %s"%str(np.mean(self.ListDictSols[iSol]["FreqDomains"],axis=1).ravel().tolist()))

            if "MSName" in self.ListDictSols[iSol].keys():
                ArrayMSNames[iFreq]=self.ListDictSols[iSol]["MSName"]

        self.NormMatrices(SolsOut.G)

        DicoOut['FreqDomains']=self.FreqDomainsOut
        DicoOut['MaskedSols']=Mask
        DicoOut['Sols']=SolsOut
        DicoOut['MSName']=ArrayMSNames
        
        self.DicoOut=DicoOut

    def Save(self,FileOut):
        if not ".npz" in FileOut: FileOut+=".npz"
        log.print("  Saving interpolated solutions in: %s"%FileOut)
        np.savez(FileOut,**self.DicoOut)


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
    if len(ListSolsFile)==0:
        raise RuntimeError("No files found")


    log.print( "Running Merge on the following SolsFile:")
    for SolsName in ListSolsFile:
        log.print( "  %s"%SolsName)
    CM=ClassMergeSols(ListSolsFile)
    CM.MakeDicoMerged()
    if options.SigmaFilterOutliers:
        CM.FilterOutliers(options.SigmaFilterOutliers)
    DicoOut=CM.Save(options.SolFileOut)


def driver():
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)
    main(options=options)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()