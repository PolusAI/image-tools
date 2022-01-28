import os
from pathlib import Path
import filepattern
import pandas as pd


import os
import filepattern 
import pprint
from pathlib import Path
import pandas as pd
from typing import List, Union, Optional, Dict
import pprint




VARIABLES = 'rtczyxp'
InpDir = '/Users/HamdahAbbasi/Desktop/test'

# def fcount(path):
#     """ Counts the number of files in a directory """
#     count = 0
#     for f in os.listdir(path):
#         if os.path.isfile(os.path.join(path, f)):
#             count += 1
            
#     return count






def chunker(sequence, chunk_size):
    it = iter(sequence)
    while True:
        chunk = []
        try:
            for i in range(chunk_size):
                chunk.append(next(it))
            yield chunk
        except StopIteration:
            if chunk:
                yield chunk
            return



# pattern='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'

# files = filepattern.FilePattern(Path(InpDir),pattern,var_order='rxytpc')



# Group_By=None

# fileslist = [file[0] for file in files(group_by=Group_By)]

# fileslist = [f for f in Path(InpDir).iterdir() if f.with_suffix('.ome.tif')]

# pf = chunker(fileslist, chunk_size=3) 
# df = []
# batch_size=[]
# for _, batch in enumerate(pf):


#     batch_files = []
#     for b in batch:
#         batch_files.append(b)
#     # pattern_regex = files.output_name(batch_files)

#     print(filepattern.infer_pattern(batch_files))

    # print(pattern_regex)


    # pattern_regex = [pattern_regex] * len(batch)
    # df.append(pattern_regex)
    # batch_size.append(len(batch))

# prf = pd.DataFrame(list(zip(df, batch_size)), columns=['FilePattren', 'Batch_Size'])

# pprint.pprint(fileslist)

  
# pattern=FilePattern


# files = filepattern.FilePattern(Path(InpDir),pattern,var_order='rxytpc')

# if pattern is None:
#     fileslist = [f for f in Path(InpDir).iterdir() if f.with_suffix('.ome.tif')]
# # else:
# #     fileslist = [file[0] for file in files(group_by=Group_By)]
# fileslist = [f for f in Path(InpDir).iterdir() if f.with_suffix('.ome.tif')]
# pf = chunker(fileslist, chunk_size=3) 
# df = []
# batch_size=[]
# for _, batch in enumerate(pf):
#     batch_files = []
#     for b in batch:
#         batch_files.append(b)

#     # if pattern is None:
#     pattern_regex = filepattern.infer_pattern(batch_files)

#     print(pattern_regex)



# fileslist = [f for f in Path(InpDir).iterdir() if f.with_suffix('.ome.tif')]
InpDir='/Users/HamdahAbbasi/Desktop/test'
Pattern='p0{r}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
OutDir='/Users/HamdahAbbasi/Desktop/test'
Var_Order='rxytpc'
# Group_By='c'



fp = filepattern.FilePattern(Path(InpDir), Pattern,var_order=Var_Order)


fileslist = [file[0] for file in fp()]

pf = chunker(fileslist, chunk_size=3) 

df = []
batch_size=[]
for _, batch in enumerate(pf):
    batch_files = []
    for b in batch:
        batch_files.append(b)  

    pattern_regex = fp.output_name(batch_files)
    df.append(pattern_regex)
    batch_size.append(len(batch))



prf = pd.DataFrame(list(zip(df, batch_size)), columns=['FilePattren', 'Batch_Size'])
prf.to_csv(os.path.join(OutDir, 'pattren_generator.csv'))
os.chdir(OutDir)
prf.to_csv('pattren_generator.csv', index=False)










    
    
    

    






    