#!/usr/bin/env python

import optparse
import sys
from Other import MyPickle
from Other import logo
from Other import ModColor
from Other import MyLogger
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

from Array import ModLinAlg

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

    SolsDico=np.load(options.SolsFile)
    Sols=SolsDico["Sols"]
    StationNames=SolsDico["StationNames"]
    Sols=Sols.view(np.recarray)
    ind=np.where(Sols.t1!=0)[0]
    Sols=Sols[ind]
    tm=(Sols.t1+Sols.t0)/2.
    t0=tm[0]
    tm-=t0

    nt,na,nd,_,_=Sols.G.shape
    nx,ny=GiveNXNYPanels(na)
    pylab.figure(1,figsize=(13,8))
    iAnt=0
    for iDir in range(nd):
        G=Sols.G[:,:,iDir,:,:]
        G=NormMatrices(G)
        ampMinMax=0,np.max(np.abs(G))
        pylab.clf()
        for i in range(nx):
            for j in range(ny):
                if iAnt>=na:continue
                J=G[:,iAnt,:,:]
                if iAnt>1:
                    ax=pylab.subplot(nx,ny,iAnt+1,sharex=ax,sharey=ax)
                else:
                    ax=pylab.subplot(nx,ny,iAnt+1)
                ax.plot(tm,np.abs(J[:,0,1]),color="gray")
                ax.plot(tm,np.abs(J[:,1,0]),color="gray")
                ax.plot(tm,np.abs(J[:,1,1]),color="black")
                ax.plot(tm,np.abs(J[:,0,0]),color="black")
                ax.set_ylim(ampMinMax)
                ax.set_xticks([])
                ax.set_yticks([])

                ax2 = ax.twinx()
                # ax.plot(tm,np.angle(J[:,0,1]),color="blue")
                # ax.plot(tm,np.angle(J[:,1,0]),color="blue")
                ax2.plot(tm,np.angle(J[:,1,1]),color="blue",alpha=0.5)
                ax2.plot(tm,np.angle(J[:,0,0]),color="blue",alpha=0.5)
                ax2.set_ylim(-np.pi,np.pi)
                ax2.set_xticks([])
                ax2.set_yticks([])
                #print StationNames[iAnt]
                pylab.title(StationNames[iAnt], fontsize=9)
                iAnt+=1
        pylab.suptitle('Direction %i'%iDir)
        pylab.tight_layout(pad=3., w_pad=0.5, h_pad=2.0)
        pylab.draw()
        pylab.show()



if __name__=="__main__":
    read_options()
    f = open(NameSave,'rb')
    options = pickle.load(f)

    main(options=options)
