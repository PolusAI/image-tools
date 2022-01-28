import os
import filepattern 
import pprint
from pathlib import Path
import pandas as pd
from typing import List, Union, Optional, Dict



class Filepattren_Generator:
    def __init__(self, 
                InpDir:Path,
                OutDir:Path,
                Pattern:Optional[str]= None,
                Chunk_Size:Optional[int]= 100,
                Var_Order:Optional[int]= None,
                Group_By: Optional[str]= None):
        self.InpDir=Path(InpDir)
        self.OutDir=Path(OutDir)
        self.Pattern=Pattern
        self.Chunk_Size=Chunk_Size
        self.Var_Order=Var_Order
        self.Group_By=Group_By
        self.fp = filepattern.FilePattern(self.InpDir, self.Pattern,var_order=self.Var_Order)
        self.fileslist = [file[0] for file in self.fp(group_by=self.Group_By)]
        if self.Pattern is None:
            self.fileslist = [f for f in self.InpDir.iterdir() if f.with_suffix('.ome.tif')]
            self.Pattern = filepattern.infer_pattern(self.fileslist)

    def batch_chunker(self):
        Iterator = iter(self.fileslist)
        while True:
            chunk = []
            try:
                for i in range(self.Chunk_Size):
                    chunk.append(next(Iterator))
                yield chunk
            except StopIteration:
                if chunk:
                    yield chunk
                return


    def pattren_generator(self):

        if self.Var_Order is None:
            self.Var_Order='rxytpc'

        pf = self.batch_chunker() 
        df = []
        batch_size=[]
        for _, batch in enumerate(pf):
            batch_files = []
            for b in batch:
                batch_files.append(b)
            pattern_regex = self.fp.output_name(batch_files)
            df.append(pattern_regex)
            batch_size.append(len(batch))

        prf = pd.DataFrame(list(zip(df, batch_size)), columns=['FilePattren', 'Batch_Size'])
        os.chdir(self.OutDir)
        prf.to_csv('pattren_generator.csv', index=False)
        return 


InpDir='/Users/HamdahAbbasi/Desktop/apply-flatfield'
Pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
OutDir='/Users/HamdahAbbasi/Desktop/test'
Var_Order='rxytpc'
Group_By='c'
dp = Filepattren_Generator(InpDir,OutDir,Pattern,100,Var_Order)


dp.pattren_generator()









# def pattren_generator(InpDir:Path,
#                     OutDir:Path,
#                     Pattern:Optional[str]= None,
#                     Var_Order:Optional[str]= None,
#                     Chunk_Size:Optional[int]= 2,
#                     Group_By: Optional[str]= None
#                     ):



#     if Pattern is None:

#         fileslist = [f for f in Path(InpDir).iterdir() if f.with_suffix('.ome.tif')]

#         Pattern = filepattern.infer_pattern(fileslist)
    

#     if Var_Order is None:
#         Var_Order='rxytpc'
    
#     fp = filepattern.FilePattern(Path(InpDir), Pattern,var_order=Var_Order)
    

#     fileslist = [file[0] for file in fp(group_by=Group_By)]

#     pf = chunker(fileslist, chunk_size=Chunk_Size) 
#     df = []
#     batch_size=[]
#     for _, batch in enumerate(pf):
#         batch_files = []
#         for b in batch:
#             batch_files.append(b)
#         pattern_regex = fp.output_name(batch_files)
#         df.append(pattern_regex)
#         batch_size.append(len(batch))

#     prf = pd.DataFrame(list(zip(df, batch_size)), columns=['FilePattren', 'Batch_Size'])
#     prf.to_csv(os.path.join(OutDir, 'pattren_generator.csv'))
#     os.chdir(OutDir)
#     prf.to_csv('pattren_generator.csv', index=False)

#     return prf


# InpDir='/Users/HamdahAbbasi/Desktop/apply-flatfield'
# Pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
# OutDir='/Users/HamdahAbbasi/Desktop/test'
# Var_Order='rxytpc'
# Group_By='c'
# prf = pattren_generator(InpDir,OutDir,Pattern,Var_Order,100)

  

   















    
    
    

    

