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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from killMS.cbuild.Gridder import _pyGridder

        
from DDFacet.Other import logger
from killMS.Other import ModColor
log=logger.getLogger("ClassWeighting")


#import ImagingWeights
from killMS.Data import ClassMS
from pyrap.tables import table

def test():
    MS=ClassMS.ClassMS("/home/bhugo/workspace/DDFworkbench/1491291289.1ghz.1.1ghz.4hrs.ms")
    ImShape=(1, 1, 375, 375)
    CellSizeRad=(1./3600)*np.pi/180
    CW=ClassWeighting(ImShape,CellSizeRad)
    flags = np.zeros((MS.uvw.shape[0], MS.ChanFreq.size,2), dtype=bool)
    WEIGHT = np.zeros((MS.uvw.shape[0], MS.ChanFreq.size), dtype=np.float32)
    CW.CalcWeights(MS.uvw, WEIGHT, flags, MS.ChanFreq)

class ClassWeighting():
    def __init__(self,
                 ImShape,
                 CellSizeRad,
                 GD=None):
        self.ImShape=ImShape
        self.CellSizeRad=CellSizeRad
        self.GD=GD
        
    def CalcWeights(self,uvw,VisWeights,flags,freqs,Robust=0,Weighting="Briggs"):
        

        #u,v,_=uvw.T

        #Robust=-2
        nch,npol,npixIm,_=self.ImShape
        FOV=self.CellSizeRad*npixIm#/2

        #cell=1.5*4./(FOV)
        cell=1./(FOV)
        #cell=4./(FOV)

        #wave=6.

        u=uvw[:,0].copy()
        v=uvw[:,1].copy()

        d=np.sqrt(u**2+v**2)
        VisWeights[d==0]=0
        Lmean=3e8/np.mean(freqs)

        uvmax=np.max(d)/Lmean#(1./self.CellSizeRad)#/2#np.max(d)
        npix=2*(int(uvmax/cell)+1)
        if (npix%2)==0:
            npix+=1

        #npix=npixIm
        xc,yc=npix//2,npix//2


        VisWeights=np.float64(VisWeights)
        #VisWeights.fill(1.)


        
        
        if Weighting=="Briggs":
            log.print( "Weighting in Briggs mode")
            log.print( "Calculating imaging weights with Robust=%3.1f on an [%i,%i] grid"%(Robust,npix,npix))
            Mode=0
        elif Weighting=="Uniform":
            log.print( "Weighting in Uniform mode")
            log.print( "Calculating imaging weights on an [%i,%i] grid"%(npix,npix))
            Mode=1
        elif Weighting=="Natural":
            log.print( "Weighting in Natural mode")
            return VisWeights
        else:
            raise ValueError("Expected Briggs, Uniform or Natural, got {0:s} for --Weighting".format(Weighting))

        grid=np.zeros((npix,npix),dtype=np.float64)


        flags=np.float32(flags)
        WW=np.mean(1.-flags,axis=2)
        VisWeights*=WW
        
        F=np.zeros(VisWeights.shape,np.int32)
        #print "u=",u
        #print "v=",v
        w=_pyGridder.pyGridderPoints(grid,
                                     F,
                                     u,
                                     v,
                                     VisWeights,
                                     float(Robust),
                                     int(Mode),
                                     np.float32(freqs.flatten()),
                                     np.array([cell,cell],np.float64))


        # C=299792458.
        # uf=u.reshape((u.size,1))*freqs.reshape((1,freqs.size))/C
        # vf=v.reshape((v.size,1))*freqs.reshape((1,freqs.size))/C

        # x,y=np.int32(np.round(uf/cell))+xc,np.int32(np.round(vf/cell))+yc
        # x,y=(uf/cell)+xc,(vf/cell)+yc
        # condx=((x>0)&(x<npix))
        # condy=((y>0)&(y<npix))
        # ind=np.where((condx & condy))[0]
        # X=x#[ind]
        # Y=y#[ind]
        
        # w[w==0]=1e-10
        
        # import pylab
        # pylab.clf()
        # #pylab.scatter(uf.flatten(),vf.flatten(),c=w.flatten(),lw=0,alpha=0.3,vmin=0,vmax=1)#,w[ind,0])
        # grid[grid==0]=1e-10
        # pylab.imshow(np.log10(grid),interpolation="nearest")
        # incr=1
        # pylab.scatter(X.ravel()[::incr],Y.ravel()[::incr],c=np.log10(w.ravel())[::incr],lw=0)#,alpha=0.3)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        
        return w



if __name__ == "__main__":
    test()