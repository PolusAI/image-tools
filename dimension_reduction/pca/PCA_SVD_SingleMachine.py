import numpy as np
import pandas as pd
from datetime import datetime
import torch
import mmap
import dask.dataframe as ddf
import dask.multiprocessing

#------------------Code Parameters------------------------------
read_option = 'direct' # 'direct' or 'mapping'
devicename = 'cpu' # 'cpu' or 'cuda:0'
applysignflip = 'yes' # 'yes' or anything else

#------------------Reading Data Directly------------------------
startTime = datetime.now()
if read_option=='direct':
#    df=pd.read_csv('/home/maghrebim2/Work/PCA_Implementations/PyTorch/10.csv')
#    d = df.values 
    df = ddf.read_csv("/home/maghrebim2/Work/PCA_Implementations/PyTorch/1200_Modified.csv",sep=',',blocksize=10000000) #dtype=types  
    d = df.compute(scheduler='processes')  #scheduler='threads' scheduler='single-threaded' scheduler='processes'
    data = np.float32(d)  
    del d
#------------------Mapping Data to Memory-----------------------
elif read_option == 'mapping':
    filename = open('/home/mahdi/Work/PCA_Implementation/PyTorch/10.csv', "r")
    m = mmap.mmap(filename.fileno(), 0, prot=mmap.PROT_READ)
    m.readline() #Remove Header Row
    linecounts = 0
    for line in filename:
        linecounts = linecounts+1
    linearray = []
    for i in range(linecounts-1):
        line = m.readline()
        linearray.append(line.strip().decode('utf-8').split(","))
    m.close()
    data = np.array(linearray,dtype='float32')
    del linearray

print ("READING Completed!")
Duration = datetime.now() - startTime
print("Duration of Reading Data == "+str(Duration))
startTime = datetime.now()
#------------------Define Device---------------------------------
device = torch.device(devicename)
print('Using device:', device)
#print(torch.cuda.current_device())
#print(torch.cuda.is_available())
#------------------Normalizing Tensor Data-----------------------
x = torch.from_numpy(data).float().to(device)
X_mean = torch.mean(x,0).to(device)
X_STD = torch.std(x,0).to(device)
x = (x-X_mean.expand_as(x))/X_STD.expand_as(x).to(device)
del X_mean
del X_STD
#------------------Compute SVD and Project Data to new PCs-------
u, s, v = torch.svd(x,some=True,compute_uv=True)
projected_data = torch.matmul(u,torch.diag(s)).to(device)
del u,s,v
#------------------Apply Sign Flip for the Projected Data--------
if applysignflip == 'yes':
    abs_projected_data=torch.abs(projected_data).to(device)
    temp = torch.eq(abs_projected_data,torch.max(abs_projected_data,-2,keepdim=True).values).type(torch.FloatTensor).to(device)
    sign_matrix = torch.sign(torch.sum(projected_data*temp,-2,keepdim=True).to(device)).to(device)
    projected_data = projected_data*sign_matrix.to(device)
#------------------Final Outputs---------------------------------
Duration = datetime.now() - startTime
print("Duration of Execution == "+str(Duration))
startTime = datetime.now()

#np.savetxt("/home/mahdi/Work/PCA_Implementation/PyCharm/PyTorch_PCA/PCA_Projected_Data.csv", projected_data.cpu().numpy(), delimiter=",")

Duration = datetime.now() - startTime
print("Duration of Writing Data == "+str(Duration))

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


