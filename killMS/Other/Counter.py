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

class Counter():
    def __init__(self,N=1):
        if N==0: N=1
        self.N=N
        self.i=-1

    def __call__(self):
        i=self.i
        #print "%i -> %i"%(i,i+1)
        self.i+=1
        Cond=(self.i%self.N==0)
        return Cond


class CounterTime():
    def __init__(self,dt=60):
        self.dt=dt
        self.CurrentTime=-1e6

    def __call__(self,time):
        rep=False
        #print time,self.CurrentTime,(time-self.CurrentTime)
        if (np.abs(time-self.CurrentTime)>self.dt):
            self.CurrentTime=time+self.dt
            rep=True
        return rep

