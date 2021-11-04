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
def ParsetToDict(fname):
    f=file(fname,"r")
    Dict={}
    ListOut=f.readlines()
    order=[]
    i=0
    for line in ListOut:
        line=line.replace("\n","")
        if not("=" in line): continue
        if "#" in line: continue
        key,val=line.split("=")
        key=key.replace(" ","")
        key=key.replace(".","_")
        #Dict[key]={"id":i,"show":1, "col":"", "help": "", "val":val}
        Dict[key]={"id":i,"val":val}
        i+=1
    return Dict

def DictToParset(Dict,fout):
     f=open(fout,"w")
     ll=sorted(Dict.items(), key=lambda x: x[1]['id'])
     Lkeys=[ll[i][0] for i in range(len(ll))]
            
     for key in Lkeys:
         keyw=key.replace("_",".")
         f.write("%s = %s\n"%(keyw,Dict[key]["val"]))
     f.close()

# def read(fin):

#     f=file(fin,"r")
#     L=f.readlines()
#     f.close()
#     D={}
#     for i in range(len(L)):
#         L[i]=L[i].replace("\n","")
#         if not("=" in L[i]): continue
#         if "#" in L[i]: continue
#         key,val=L[i].split("=")
#         D[key]=val
#     return D


