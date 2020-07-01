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
def reformat(ssin,slash=True,LastSlash=True):
    ss=ssin.split("/")
    ss=filter (lambda a: a != "", ss)
    ss="/".join(ss)+"/"
    if ssin[0]=="/": ss="/"+ss
    if not(slash):
        ss=ss[1:-1]
    if not(LastSlash):
        if ss[-1]=="/": ss=ss[0:-1]
    return ss
