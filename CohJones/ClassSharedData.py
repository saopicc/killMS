import numpy as np
import NpShared

class ClassSharedData():
    def __init__(self,PrefixShared="Default"):


        self.PrefixShared=PrefixShared
        

    def setData(self):
        DATA={}
        Fields=['freqs', 'times', 'A1', 'A0', 'flags', 'uvw', 'data']
        for key in Fields:
            Name="%s.%s"%(self.PrefixShared,key)
            DATA[key]=NpShared.GiveArray(Name)
        self.DATA=DATA
