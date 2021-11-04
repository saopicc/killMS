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
from SkyModel.Sky import ClassSM
from killMS.Other import rad2hmsdms
import os
from killMS.Data import ClassMS
import ephem
from killMS.Other import ModParsetType
from pyrap.tables import table

MSTemplate="/media/tasse/data/MS/0000.MS"
WorkingDir="/media/tasse/data/MS/"
#ProgTables="/home/tasse/sources/LOFAR/build/gnu_opt/LCS/MSLofar/src/makebeamtables"
ProgTables="makebeamtables"


# MSTemplate="L102479_SB144_uv.dppp.MS.dppp.tsel_fixed"
# WorkingDir="/data/tasse/Simul2/"
# ProgTables="makebeamtables"

# antennaset="LBA_INNER"
# #antennaset="HBA_INNER"
# StaticMetaDataDir="/home/cyril.tasse/source/LOFARBeamData"

StaticMetaDataDir="/media/tasse/data/VE_Py3_bis/sources/lofar/MAC/Deployment/data/StaticMetaData"
AntennaSets="/media/tasse/data/VE_Py3_bis/sources/lofar/MAC/Deployment/data/StaticMetaData/AntennaSets.conf"

def test():

    # Random catalog
    DicoPropPointings={}

    # DicoPropPointings[0]={"offset":(0,0),
    #                       "Ns":10,
    #                       "Nc":2,
    #                       "Diam":1,
    #                       #"SI":np.array([100]*10,dtype=np.float32),
    #                       "finfo":(100e6,250e6,10)}
    # DicoPropPointings[1]={"offset":(5,5),
    #                       "Ns":9,
    #                       "Nc":3,
    #                       "Diam":1,
    #                       "finfo":(100e6,250e6,10)}

    DicoPropPointings[0]={"offset":(0,0),
                          "Ns":9,
                          "Nc":0,
                          "Diam":4,
                          "finfo":(40e6,60e6,2),
                          "Mode":"Grid"}

    DicoPropPointings[1]={"offset":(10,5),
                          "Ns":9,
                          "Nc":0,
                          "Diam":4,
                          "finfo":(40e6,60e6,2),
                          "Mode":"Grid"}

    # DicoPropPointings[0]={"offset":(0,0),
    #                       "Ns":,
    #                       "Nc":0,
    #                       "Diam":0.5,
    #                       "finfo":(50e6,250e6,1),
    #                       "Mode":"Grid"}

    # DicoPropPointings[0]={"offset":(0,0),
    #                       "Ns":100,
    #                       "Nc":10,
    #                       "Diam":1,
    #                       "finfo":(50e6,250e6,1),
    #                       "Mode":"Random"}


    # # Single source
    # DicoPropPointings={}
    # DicoPropPointings[0]={"offset":(0,0),
    #                       "Ns":-1,
    #                       "Nc":0,
    #                       "Diam":1,
    #                       "SI":np.array([100],dtype=np.float32),
    #                       "finfo":(70e6,150e6,10)}
    # DicoPropPointings[1]={"offset":(5,5),
    #                       "Ns":-1,
    #                       "Nc":0,
    #                       "Diam":1,
    #                       "SI":np.array([100],dtype=np.float32),
    #                       "finfo":(70e6,150e6,10)}
    
    ObsMachine=MakeMultipleObs(DicoPropPointings)
    ObsMachine.MakeSM_MS0()
    ObsMachine.DuplicateMSInFreq()
    ObsMachine.MakeClusterCats()

def GiveDate(tt):
    import pyrap.quanta as qa
    import pyrap.measures as pm
    time_start = qa.quantity(tt, 's')
    me = pm.measures()
    dict_time_start_MDJ = me.epoch('utc', time_start)
    time_start_MDJ=dict_time_start_MDJ['m0']['value']
    JD=time_start_MDJ+2400000.5-2415020
    d=ephem.Date(JD)
    return d.datetime().isoformat().replace("T","/")

def BBSprintRandomSM(Ns,Ddeg,ra_mean_dec_mean,OutFile="ModelRandom0",ra_dec_offset=(0,0),SI=None,Mode="Random"):
    ra_mean,dec_mean=ra_mean_dec_mean
    ang=Ddeg*np.pi/180
    if Ns!=-1:
        if Mode=="Random":
            ra=ra_mean+ang*np.random.randn(Ns)
            dec=dec_mean+ang*np.random.randn(Ns)
        elif Mode=="Grid":
            Ns=int(np.sqrt(Ns))
            ra,dec=np.mgrid[-ang:ang:Ns*1j,-ang:ang:Ns*1j]
            ra=ra_mean+ra.flatten()
            dec=dec_mean+dec.flatten()
            Ns=ra.size
    else:
        ra=np.array([ra_mean])
        dec=np.array([dec_mean])
        Ns=1

    Cat=np.zeros((Ns,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                ('Gmaj',np.float),('Gangle',np.float),("Select",np.int)])
    Cat=Cat.view(np.recarray)

    #SM=ClassSM("/media/tasse/data/HYPERCAL/test/ModelIon.txt")
    #ra_mean  = np.mean(SM.SourceCat.ra)+ra_dec_offset[0]*np.pi/180
    #dec_mean = np.mean(SM.SourceCat.dec)+ra_dec_offset[1]*np.pi/180

    Cat.ra=ra
    Cat.dec=dec
    Cat.I=np.random.rand(Ns)
    Cat.alpha=-np.random.rand(Ns)
    Cat.alpha=0.
    #Cat.I[0]=0
    Cat.I.fill(1)
    Cat.I/=np.sum(Cat.I)
    #Cat.I*=100
    if SI!=None:
        Cat.I=SI
        Cat.Sref=SI
    
    WriteBBSCat(OutFile,Cat)

def WriteBBSCat(OutFile,Cat):
    Ns=Cat.shape[0]

    f = open(OutFile, 'w')
    Names=["%3.3i"%i for i in range(Ns)]
    ss="# (Name, Type, Ra, Dec, I, Q, U, V, ReferenceFrequency='7.38000e+07', SpectralIndex='[]', MajorAxis, MinorAxis, Orientation) = format"
    f.write(ss+'\n')
    for i in range(Ns):
        SRa=rad2hmsdms.rad2hmsdms(Cat.ra[i],Type="ra").replace(" ",":")
        SDec=rad2hmsdms.rad2hmsdms(Cat.dec[i]).replace(" ",".")
        sI=str(Cat.I[i])
        sAlpha=str(Cat.alpha[i])#0#str(np.random.randn(1)[0]*0.2-0.8)
        ss="%s, POINT, %s,  %s, %s, 0.0, 0.0, 0.0, 7.38000e+07, [%s], 0, 0.00000e+00, 0.0"%(Names[i],SRa,SDec,sI,sAlpha)
        print(ss)
        f.write(ss+'\n')        
    f.close()


class MakeMultipleObs():

    def __init__(self,DicoPropPointings,
                 MSTemplateName=MSTemplate,
                 BaseDir=WorkingDir
    ):
        self.DicoMS={}
        self.MSTemplateName=MSTemplateName
        self.MSTemplate=ClassMS.ClassMS(MSTemplateName,DoReadData=False)

        self.BaseDir=BaseDir
        self.DicoPropPointings=DicoPropPointings

    def MakeClusterCats(self):
        sCat=[]
        sSM=[]
        for key in sorted(self.DicoPropPointings.keys()):
            D=self.DicoPropPointings[key]
            _,_,nf=D["finfo"]
            ListMS=D["ListMS"]
            Cat=np.zeros((nf,),dtype=[('node', 'S200'), ('dirMSname', 'S200'), ('PointingID', int)])
            Cat=Cat.view(np.recarray)
            Cat.node="igor"
            Cat.dirMSname[:]=ListMS[:]
            Cat.PointingID[:]=key
            CatName=D["SMName"]+".cluster.npy"
            np.save(CatName,Cat)
            sCat.append(CatName)
            sSM.append(D["SM"])

        print(", ".join(sCat))
        print(", ".join(sSM))
        

    def MakeSM_MS0(self):

        for key in sorted(self.DicoPropPointings.keys()):
            D=self.DicoPropPointings[key]

            SMName="ModelRandom%2.2i.txt"%key
            D["SMName"]=SMName
            offset,Ns,Nc,Diam,Mode=D["offset"],D["Ns"],D["Nc"],D["Diam"],D["Mode"]
            rac,decc=self.MSTemplate.radec
            rac+=offset[0]*np.pi/180
            decc+=offset[1]*np.pi/180
            SI=None
            if "SI" in D.keys():
                SI=D["SI"]
            BBSprintRandomSM(Ns,Diam,(rac,decc),OutFile=SMName,ra_dec_offset=(0,0),SI=SI,Mode=Mode)
            sExec="MakeModel.py --SkyModel=%s --NCluster=%i --CMethod=4 --DoPlot=0"%(SMName,Nc)
            print(sExec)
            os.system(sExec)

            ModelName=SMName+".npy"

            D["SM"]=ModelName
            DirName="%sPointing%2.2i/"%(self.BaseDir,key)
            os.system("rm -rf %s"%DirName)
            ss="mkdir -p %s"%DirName
            os.system(ss)

            D["dirName"]="%s"%(DirName)
            D["MS0Name"]="Template_%4.4i.MS"%(0)
            D["dirMS0Name"]="%s%s"%(DirName,D["MS0Name"])
            
            self.MakeMS(D["dirMS0Name"],(rac,decc))
            D["dirMS0Name"]+="_p0"

            self.setAntNames(D["dirMS0Name"])
            

            # #os.system("cp -r %s/LOFAR_* %s"%(self.MSTemplateName,self.DicoMS["MS0Name"]))
            # ss="%s "%ProgTables + "antennafielddir=%s/AntennaFields "%StaticMetaDataDir + "antennaset=%s antennasetfile=%s/AntennaSets.conf "%(antennaset,StaticMetaDataDir) + "ihbadeltadir=%s/iHBADeltas "%StaticMetaDataDir + "ms=%s overwrite=1"%(D["dirMS0Name"])
            # print(ss)
            # os.system(ss)

            #ss="%s antennafielddir=/home/tasse/sources/StaticMetaData antennaset=LBA_INNER antennasetfile=/home/tasse/sources/AntennaSets.conf ihbadeltadir=/home/tasse/sources/StaticMetaData ms=%s overwrite=1"%(ProgTables,D["dirMS0Name"])


            ss="%s antennaset=LBA_INNER ms=%s overwrite=1"%(ProgTables,D["dirMS0Name"])
            #ss="%s antennaset=LBA_INNER ms=%s antennasetfile=%s overwrite=1"%(ProgTables,D["dirMS0Name"],AntennaSets)
            ss="%s antennaset=LBA_INNER ms=%s antennasetfile=%s/AntennaSets.conf antennafielddir=%s/AntennaFields overwrite=1"%(ProgTables,D["dirMS0Name"],StaticMetaDataDir,StaticMetaDataDir)

#            ss="%s antennafielddir=/home/cyril.tasse/source/LOFARBeamData/AntennaFields antennaset=LBA_INNER antennasetfile=/home/cyril.tasse/source/LOFARBeamData/AntennaSets.conf ihbadeltadir=/home/cyril.tasse/source/LOFARBeamData/iHBADeltas ms=%s overwrite=1"%(ProgTables,D["dirMS0Name"])
            print(ss)
            os.system(ss)
            
# makebeamtables antennafielddir=/home/cyril.tasse/source/LOFARBeamData/AntennaFields antennaset=LBA_INNER antennasetfile=/home/cyril.tasse/source/LOFARBeamData/AntennaSets.conf ihbadeltadir=/home/cyril.tasse/source/LOFARBeamData/iHBADeltas ms=/data/tasse/Simul/Pointing00/Template_0000.MS_p0 overwrite=1
# makebeamtables antennafielddir=/home/cyril.tasse/source/StaticMetaData/AntennaFields antennaset=LBA_INNER antennasetfile=/home/cyril.tasse/source/StaticMetaData/AntennaSets.conf ihbadeltadir=/home/cyril.tasse/source/StaticMetaData/iHBADeltas ms=/data/tasse/Simul/Pointing00/Template_0000.MS_p0 overwrite=1

    def setAntNames(self,MSName):
        t0=table(self.MSTemplate.MSName+"/ANTENNA",readonly=False)
        t1=table(MSName+"/ANTENNA",readonly=False)
        t1.putcol("NAME",t0.getcol("NAME"))
        t0.close()
        t1.close()



    def MakeMS(self,MSName,ra_dec):
        ra,dec=ra_dec
        D={}
        MS=self.MSTemplate

        StrRA  = rad2hmsdms.rad2hmsdms(ra,Type="ra").replace(" ",":")
        StrDEC = rad2hmsdms.rad2hmsdms(dec,Type="dec").replace(" ",".")

        DateTime=GiveDate(np.min(MS.F_times))
        # Date,Time=DateTime.datetime().date().isoformat(),DateTime.datetime().time().isoformat()
        # SDateTime=Date+'/'+Time

        D["AntennaTableName"]={"id":0,"val":self.MSTemplate.MSName+"/ANTENNA"}
        D["Declination"]={"id":0,"val":StrDEC}
        D["RightAscension"]={"id":0,"val":StrRA}
        D["MSName"]={"id":0,"val":MSName}
        D["NBands"]={"id":0,"val":1}
        D["WriteAutoCorr"]={"id":0,"val":"T"}

        D["NFrequencies"]={"id":0,"val":4} # MS.Nchan}
        D["StepFreq"]={"id":0,"val":10e6} # np.abs(self.MSTemplate.dFreq)}

        D["StartFreq"]={"id":0,"val":np.min(self.MSTemplate.ChanFreq.flatten())-np.abs(self.MSTemplate.dFreq[0])/2.}
        D["StartTime"]={"id":0,"val":DateTime}

        D["StepTime"]={"id":0,"val":30*5}#self.MSTemplate.dt}#MS.dt}
        D["NTimes"]={"id":0,"val":300}#int((np.max(self.MSTemplate.F_times)-np.min(self.MSTemplate.F_times))/self.MSTemplate.dt)}
        #D["NTimes"]={"id":0,"val":int((np.max(self.MSTemplate.F_times)-np.min(self.MSTemplate.F_times))/self.MSTemplate.dt)}
        
        D["NParts"]={"id":0,"val":"1"}

        D["VDSPath"]={"id":0,"val":"."}
        D["WriteImagerColumns"]={"id":0,"val":"T"}

        ModParsetType.DictToParset(D,"makems.tmp.cfg")
        
        os.system("cat makems.tmp.cfg")
        os.system("makems makems.tmp.cfg")

    def DuplicateMSInFreq(self):
        for key in sorted(self.DicoPropPointings.keys()):
            D=self.DicoPropPointings[key]
            fmin,fmax,n=D["finfo"]

            freqs=np.linspace(fmin,fmax,n)
            iSB=0
            #os.system("mkdir -p /media/6B5E-87D0/MS/SimulTec/many")
            #os.system("rm -Rf /media/6B5E-87D0/MS/SimulTec/many/*")
            dirMS="%sMS/"%(D["dirName"])
            os.system("mkdir %s"%dirMS)
            dirMS0Name=D["dirMS0Name"]
            D["ListMS"]=[]
            for freq in freqs:
                #outn="%s%4.4i.MS"%(dirMS,iSB)
                outn="%4.4i.p%2.2i.MS"%(iSB,int(key))
                D["ListMS"].append(outn)

                ss="cp -r %s %s "%(dirMS0Name,outn)
                print("make ",outn,", at f=",freq)
                print("       %s"%ss)
                #os.system("cp -r /media/6B5E-87D0/MS/SimulTec/Simul_one.beam_off.gauss.MS.tsel "+outn)
                os.system(ss)
                ta_spectral=table(outn+'/SPECTRAL_WINDOW/',ack=False,readonly=False)

                dummy=ta_spectral.getcol('REF_FREQUENCY')
                dummy=dummy-np.mean(dummy)+freq
                
                #dummy.fill(freq)
                ta_spectral.putcol('REF_FREQUENCY',dummy)

                dummy=ta_spectral.getcol('CHAN_FREQ')
                dummy=dummy-np.mean(dummy)+freq
                #dummy.fill(freq)
                ta_spectral.putcol('CHAN_FREQ',dummy)
                ta_spectral.close()
                iSB+=1


def MakeClusterCat():
    ParsetFile="ParsetNew.txt"
    GD=ClassGlobalData(ParsetFile)
    
    
    
