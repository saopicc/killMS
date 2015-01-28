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
     ll=sorted(Dict.iteritems(), key=lambda x: x[1]['id'])
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


