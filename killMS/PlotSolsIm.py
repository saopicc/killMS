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
import sys
from killMS.Other import MyPickle
from killMS.Other import logo
from killMS.Other import ModColor
from DDFacet.Other import logger
import matplotlib.gridspec as gridspec
log=logger.getLogger("killMS")
#logger.itsLog.logger.setLevel(logger.logging.CRITICAL)
from itertools import product as ItP

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]

# test

#import numpy
#print numpy.__file__
#import pyrap
#print pyrap.__file__
#stop


if "nocol" in sys.argv:
    print("nocol")
    ModColor.silent=1
if "nox" in sys.argv:
    import matplotlib
    matplotlib.use('agg')
    print(ModColor.Str(" == !NOX! =="))

import time
import os
import numpy as np
import pickle

NameSave="last_plotSols.obj"
def read_options():

    desc="""killMS Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SolsFile',help='Input Solutions list [no default]',default='')
    group.add_option('--DoResid',type="int",help='No [no default]',default=-1)
    group.add_option('--PlotMode',type='int',help=' [no default]',default=0)
    group.add_option('--DirList',help=' [no default]',default="")
    group.add_option('--FlagStations',type=str,help=' [no default]',default="")
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open(NameSave,"wb")
    pickle.dump(options,f)
    
import pylab

import numpy as np

def GiveNXNYPanels(Ns,ratio=800/500):
    nx=int(round(np.sqrt(Ns/ratio)))
    ny=int(nx*ratio)
    if nx*ny<Ns: ny+=1
    return nx,ny

from killMS.Array import ModLinAlg

def NormMatrices(G):
    print("no norm")
    return G
    nt,nch,na,_,_=G.shape

    for iChan,it in ItP(range(nch),range(nt)):
        Gt=G[it,iChan,:,:]
        u,s,v=np.linalg.svd(Gt[0])
        # #J0/=np.linalg.det(J0)
        # J0=Gt[0]
        # JJ=np.dot(J0.T.conj(),J0)
        # sqJJ=ModLinAlg.sqrtSVD(JJ)
        # sqJJinv=ModLinAlg.invSVD(JJ)
        # U=np.dot(J0,sqJJinv)
        U=np.dot(u,v)
        for iAnt in range(0,na):
            Gt[iAnt,:,:]=np.dot(U.T.conj(),Gt[iAnt,:,:])
            #Gt[iAnt,:,:]=np.dot(np.dot(u,Gt[iAnt,:,:]),v.T.conj())
            #Gt[iAnt,:,:]=np.dot(Gt[iAnt,:,:],J0)
    return G

# def NormMatrices(G):
#     nt,na,_,_=G.shape
#     nt,nch,na,nd,_,_=LSols[0].G.shape
#     for it in range(nt):
#         Gt=G[it,:,:,:]
#         u,s,v=np.linalg.svd(Gt[0])
#         # #J0/=np.linalg.det(J0)
#         # J0=Gt[0]
#         # JJ=np.dot(J0.T.conj(),J0)
#         # sqJJ=ModLinAlg.sqrtSVD(JJ)
#         # sqJJinv=ModLinAlg.invSVD(JJ)
#         # U=np.dot(J0,sqJJinv)
#         U=np.dot(u,v)
#         for iAnt in range(0,na):
#             Gt[iAnt,:,:]=np.dot(U.T.conj(),Gt[iAnt,:,:])
#             #Gt[iAnt,:,:]=np.dot(np.dot(u,Gt[iAnt,:,:]),v.T.conj())
#             #Gt[iAnt,:,:]=np.dot(Gt[iAnt,:,:],J0)
#     return G


def main(options=None):
    if options is None:
        f = open(NameSave,'rb')
        options = pickle.load(f)
    PM=PlotMachine(options)
    PM.PlotAll()
    
class PlotMachine():
    def __init__(self,options):
        FilesList=options.SolsFile.split(",")
        LSols=[]
        nSol=len(FilesList)
        t0=None
        for FileName in FilesList:
            SolsDico=dict(np.load(FileName,allow_pickle=True))
            Sols=SolsDico["Sols"]
            Sols=Sols.view(np.recarray)
            
            ind=np.where(Sols.t1!=0)[0]
            Sols=Sols[ind]
            tm=(Sols.t1+Sols.t0)/2.
            if t0==None:
                t0=tm[0]
            tm-=t0
            Sols.t0=tm
            LSols.append(Sols)
            StationNames=SolsDico["StationNames"]

            if options.FlagStations!="":
                indStations=np.arange(len(StationNames))
                StationNamesSel=[]
                for iAnt,Name in enumerate(StationNames):
                    if options.FlagStations in Name:
                        indStations[iAnt]=-1
                        log.print("Flagging station %s"%Name)
                    else:
                        StationNamesSel.append(Name)
                indStations=indStations[indStations!=-1]
                nas=indStations.size
                nt,nch,na,nd,_,_=LSols[-1].G.shape
                SolsOut=np.zeros((nt,),dtype=[("t0",np.float64),("t1",np.float64),
                                              ("G",np.complex64,(nch,nas,nd,2,2)),
                                              ("Stats",np.float32,(nch,nas,4))])
                SolsOut=SolsOut.view(np.recarray)
                SolsOut.t0=Sols.t0
                SolsOut.t1=Sols.t1
                SolsOut.G=Sols.G[:,:,indStations,:,:,:]
                LSols[-1]=SolsOut
                SolsDico["StationNames"]=StationNamesSel
                StationNames=SolsDico["StationNames"]
                if 'MaskedSols' in SolsDico.keys():
                    SolsDico['MaskedSols']=SolsDico['MaskedSols'][:,:,indStations,:,:,:]

            SolsOut=LSols[-1]
            if 'MaskedSols' in SolsDico.keys():
                M=SolsDico['MaskedSols'][:,:,:,0,0,0]
                Mfreq=(np.sum(np.sum(M,axis=0),axis=1)==0)
                Sols=LSols[-1]
                nt,nch,nas,nd,_,_=LSols[-1].G.shape
                nch=np.count_nonzero(Mfreq)
                SolsOut=np.zeros((nt,),dtype=[("t0",np.float64),("t1",np.float64),
                                              ("G",np.complex64,(nch,nas,nd,2,2)),
                                              ("Stats",np.float32,(nch,nas,4))])
                SolsOut=SolsOut.view(np.recarray)
                SolsOut.t0=Sols.t0
                SolsOut.t1=Sols.t1
                SolsOut.G=Sols.G[:,Mfreq,:,:,:,:]
                LSols[-1]=SolsOut


                
                print(SolsOut.G.shape)
                
        nt,nch,na,nd,_,_=LSols[0].G.shape
        if options.DoResid!=-1:
            Sresid=LSols[1].copy()
            LSols.append(Sresid)
    
        if options.DirList!="":
            DirList=options.DirList.split(',')
            DirList=[int(i) for i in DirList]
        else:
            DirList=range(nd)
    
        # #fig.subplots_adjust(wspace=0, hspace=0)
        # for iDir in DirList:
        #     for iSol in range(nSol):
        #         Sols=LSols[iSol]
        #         G=Sols.G[:,:,iDir,:,:]
        #         Sols.G[:,:,iDir,:,:]=NormMatrices(G)

        self.Mask=None
        if 'MaskedSols' in SolsDico.keys():
            log.print("Some solutions are masked")
            M=SolsDico['MaskedSols']
            # Sols.G[M==1]=np.nan
            self.Mask=M
    
        ampMax=1.5*np.max(np.median(np.abs(LSols[0].G),axis=1))
        if options.PlotMode==0:
            op0=np.abs
        else:
            op0=np.angle
    
        ylim0=0,len(DirList)
            
        # if options.DoResid!=-1:
        #     LSols[-1].G[:,:,iDir,:,:]=LSols[1].G[:,:,iDir,:,:]-LSols[0].G[:,:,iDir,:,:]
        #     nSol+=1
        self.DirList=DirList
        self.LSols=LSols

    def PlotAll(self):
        for iDir in self.DirList:
            self.Plot(self.LSols,iDir)
    
    def Plot(self,LSols,iDir=0):
        print(iDir)
        op0=np.abs
        op1=np.angle
        #op0=np.real
        #op1=np.imag
        
        nt,nch,na,nd,_,_=LSols[0].G.shape
        nx,ny=GiveNXNYPanels(na)
        fig=pylab.figure(0,figsize=(13,8))
        gs1 = gridspec.GridSpec(2*nx, ny)
        gs1.update(wspace=0.05, hspace=0.05, left=0.05, right=0.95, bottom=0.05, top=0.95)
    
        pylab.clf()
    
        if len(LSols)==1:
            Sols=LSols[0]
            ADir_0=op0(Sols.G[:,:,:,iDir,0,0])
            ADir_1=op1(Sols.G[:,:,:,iDir,0,0])
            vmin,vmax=0,2
            
        if len(LSols)==2:
            Sols=LSols[0]
            ADir_0=op0(Sols.G[:,:,:,iDir,0,0])-op0(LSols[1].G[:,:,:,iDir,0,0])
            ADir_1=op1(Sols.G[:,:,:,iDir,0,0]*LSols[1].G[:,:,:,iDir,0,0].conj())
    
        Mean=np.mean(ADir_0)
        MAD=np.sqrt(np.median((ADir_0-Mean)**2))
        vmin,vmax=Mean-10*MAD,Mean+10*MAD
    
        # if self.Mask is not None:
        #     M=self.Mask[:,:,:,iDir,0,0]
        #     ADir_0[M==1]=np.nan
        #     ADir_1[M==1]=np.nan


            
        iAnt=0
        for i in range(nx):
            for j in range(ny):
                if iAnt>=na:continue
                #pylab.title(StationNames[iAnt], fontsize=9)
    
                A_0=ADir_0[:,:,iAnt]
                A_1=ADir_1[:,:,iAnt]
                ax = pylab.subplot(gs1[2*i,j])
                #ax2 = ax.twinx()
                ax.imshow(A_0.T,vmin=vmin,vmax=vmax,interpolation="nearest",aspect='auto',cmap="gray")
                nt,nch,na,nd,_,_=Sols.G.shape
                ax.set_xticks([])
                ax.set_yticks([])
                
                ax = pylab.subplot(gs1[2*i+1,j])
                #ax2 = ax.twinx()
                #A=Sols.G[:,:,iAnt,iDir,0,0]
                F=5.
                ax.imshow(A_1.T,vmin=-np.pi/F,vmax=np.pi/F,interpolation="nearest",aspect='auto')
                #ax.imshow(A_1.T,interpolation="nearest",aspect='auto')
                #print iAnt,np.max(np.abs(A_1))
                nt,nch,na,nd,_,_=Sols.G.shape
                ax.set_xticks([])
                ax.set_yticks([])
    
    
    
                    
                iAnt+=1
        pylab.suptitle('Direction %i (op0[%5.2f, %5.2f])'%(iDir,vmin,vmax))
        #pylab.tight_layout(pad=3., w_pad=0.5, h_pad=2.0)
        pylab.draw()
        pylab.show()
        #pylab.pause(0.1)
        #time.sleep(1)


def driver():
    read_options()
    f = open(NameSave,'rb')
    options = pickle.load(f)

    PM=PlotMachine(options=options)
    PM.PlotAll()

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()