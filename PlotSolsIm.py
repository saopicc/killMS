#!/usr/bin/env python

import optparse
import sys
from killMS2.Other import MyPickle
from killMS2.Other import logo
from killMS2.Other import ModColor
from killMS2.Other import MyLogger
log=MyLogger.getLogger("killMS")
MyLogger.itsLog.logger.setLevel(MyLogger.logging.CRITICAL)

sys.path=[name for name in sys.path if not(("pyrap" in name)&("/usr/local/lib/" in name))]

# test

#import numpy
#print numpy.__file__
#import pyrap
#print pyrap.__file__
#stop


if "nocol" in sys.argv:
    print "nocol"
    ModColor.silent=1
if "nox" in sys.argv:
    import matplotlib
    matplotlib.use('agg')
    print ModColor.Str(" == !NOX! ==")

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

from killMS2.Array import ModLinAlg

def NormMatrices(G):
    nt,na,_,_=G.shape

    for it in range(nt):
        Gt=G[it,:,:,:]
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


def main(options=None):
    

    if options==None:
        f = open(NameSave,'rb')
        options = pickle.load(f)



    FilesList=options.SolsFile.split(",")
    LSols=[]
    nSol=len(FilesList)
    t0=None
    for FileName in FilesList:
        SolsDico=np.load(FileName)
        Sols=SolsDico["Sols"]
        Sols=Sols.view(np.recarray)

        ind=np.where(Sols.t1!=0)[0]
        Sols=Sols[ind]
        tm=(Sols.t1+Sols.t0)/2.
        if t0==None:
            t0=tm[0]
        tm-=t0
        Sols.t0=tm
        nt,na,nd,_,_=Sols.G.shape
        nx,ny=GiveNXNYPanels(na)
        LSols.append(Sols)
        StationNames=SolsDico["StationNames"]

        # LSols=[LSols[0]]
        # nSol=1

    # diag terms
    Lls=["-","--",":"]
    Lcol0=["black","black","blue"]
    Lcol1=["gray","gray","red"]
    Lalpha0=[1,1,1]
    Lalpha1=[0.5,0.5,0.5]
    # off-diag terms
    Lls_off=["-","--",":"]
    Lcol0_off=["black","black","blue"]
    Lcol1_off=["gray","gray","red"]
    
    if options.DoResid!=-1:
        Sresid=LSols[1].copy()
        LSols.append(Sresid)

    if options.DirList!="":
        DirList=options.DirList.split(',')
        DirList=[int(i) for i in DirList]
    else:
        DirList=range(nd)

    for iDir in DirList:
        pylab.figure(0,figsize=(13,8))
        iAnt=0
        for iSol in range(nSol):
            Sols=LSols[iSol]
            G=Sols.G[:,:,iDir,:,:]
            Sols.G[:,:,iDir,:,:]=NormMatrices(G)
            
        ampMax=1.5*np.max(np.median(np.abs(LSols[0].G),axis=1))
        if options.PlotMode==0:
            op0=np.abs
            op1=np.angle
            ylim0=0,ampMax
            ylim1=-np.pi,np.pi
        else:
            op0=np.real
            op1=np.imag
            ylim0=-ampMax,ampMax
            ylim1=-ampMax,ampMax

        if options.DoResid!=-1:
            LSols[-1].G[:,:,iDir,:,:]=LSols[1].G[:,:,iDir,:,:]-LSols[0].G[:,:,iDir,:,:]
            nSol+=1
            

        pylab.clf()

        for i in range(nx):
            for j in range(ny):
                if iAnt>=na:continue
                if iAnt>=1:
                    ax=pylab.subplot(nx,ny,iAnt+1,sharex=axRef,sharey=axRef)
                else:
                    axRef=pylab.subplot(nx,ny,iAnt+1)
                    ax=axRef
                ax2 = ax.twinx()
                pylab.title(StationNames[iAnt], fontsize=9)
                for iSol in [0]:
                    Sols=LSols[iSol]
                    ax.imshow(np.abs(Sols.G[:,iAnt,:,0,0]).T,vmin=0,vmax=2,interpolation="nearest")

                    #ax.set_ylim(ylim0)
                    ax.set_xticks([])
                    ax.set_yticks([])
    
                iAnt+=1
        pylab.suptitle('Direction %i'%iDir)
        pylab.tight_layout(pad=3., w_pad=0.5, h_pad=2.0)
        pylab.draw()
        pylab.show()
        #pylab.pause(0.1)
        #time.sleep(1)


if __name__=="__main__":
    read_options()
    f = open(NameSave,'rb')
    options = pickle.load(f)

    main(options=options)
