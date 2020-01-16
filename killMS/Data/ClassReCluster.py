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
from DDFacet.Other import logger
log=logger.getLogger("ClassReCluster")
from killMS.Other import ModColor
from killMS.Other import reformat
import os

class ClassReCluster():
    def __init__(self,GD):
        self.GD=GD

    def ReClusterSkyModel(self,SM,MSName):
        SolRefFile=self.GD["PreApply"]["PreApplySols"][0]

        if (SolRefFile!="")&(not(".npz" in SolRefFile)):
            Method=SolRefFile
            ThisMSName=reformat.reformat(os.path.abspath(MSName),LastSlash=False)
            SolRefFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)


        log.print( ModColor.Str("Re-clustering input SkyModel to match %s clustering"%SolRefFile))
        
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
        #print SM.SourceCat.Cluster
        SM.SourceCat.Cluster[:]=Cluster[:]
        SM.ClusterCat=ClusterCat0
        #print SM.SourceCat.Cluster
        
        SM.Dirs=sorted(list(set(SM.SourceCat.Cluster.tolist())))
        SM.NDir=len(SM.Dirs)

        log.print( "  There are %i clusters in the re-clustered skymodel"%SM.NDir)

        NDir=lc.size
        for iDir in range(NDir):
            ind=(SM.SourceCat.Cluster==iDir)
            SM.ClusterCat.SumI[iDir]=np.sum(SM.SourceCat.I[ind])

