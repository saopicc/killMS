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

import time as timemod
from DDFacet.Other import logger
log=logger.getLogger("ClassTimeIt")
DoLog=False


class ClassTimeIt():
    def __init__(self,name="",f=1.):
        self.t0=timemod.time()
        self.IsEnable=True
        if name=="":
            self.name=name
        else:
            self.name=name+": "
        self.IsEnableIncr=False
        self.Counter=""
        self.f=f

    def reinit(self):
        self.t0=timemod.time()

    def timeit(self,stri=" Time",hms=False):
        if self.IsEnable==False: return
        t1=timemod.time()
        dt=self.f*(t1-self.t0)
        if not(hms):
            Sout= "  * %s%s %s : %7.5fs"%(self.name,stri,str(self.Counter),dt)
            if self.IsEnableIncr: self.Counter+=1
        else:
            ss=(dt)/60.
            m=int(ss)
            s=(ss-m)*60.
            Sout= "  * %s computation time: %i min. %4.1f sec."%(stri,m,s)
        self.t0=t1
        if DoLog:
            log.print( Sout)
        else:
            print(Sout)

        return dt

    def timeitHMS(self,stri=" Time"):
        t1=timemod.time()
        self.t0=t1

    def disable(self):
        self.IsEnable=False

    def enableIncr(self,incr=1):
        self.IsEnableIncr=True
        self.Counter=0

    def AddDt(self,Var):
        t1=timemod.time()
        dt=t1-self.t0
        Var=Var+dt
        self.t0=t1
        return Var
