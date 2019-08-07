import numpy as np
import pandas as pd
from datetime import datetime
import torch
import mmap
import dask.dataframe as ddf
import dask.multiprocessing
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('readOption', type=str)
argparser.add_argument('deviceName', type=str)
argparser.add_argument('applySignFlip', type=str)
argparser.add_argument('inputPath', type=str)
argparser.add_argument('outputPath', type=str, nargs='?', default='./PCA_Projected_Data_Final.csv')
args = argparser.parse_args()

startTime = datetime.now()
#Reading input data directly and creating a numpy array.
if args.readOption=='direct':
    df = ddf.read_csv(args.inputPath,sep=',',blocksize=10000000)   
    d = df.compute(scheduler='threads') 
    data = np.float32(d)  
    del d

#Mapping Data to Memory and create a numpy array.
#This method is good if data cannot be fit into the memory.
elif args.readOption == 'mapping':
    fileName = open(args.inputPath, "r")
    m = mmap.mmap(fileName.fileno(), 0, prot=mmap.PROT_READ)
    #Remove Header Row
    m.readline() 
    lineCounts = 0
    for line in fileName:
        lineCounts = lineCounts+1
    lineArray = []
    for i in range(lineCounts-1):
        line = m.readline()
        lineArray.append(line.strip().decode('utf-8').split(","))
    m.close()
    data = np.array(lineArray,dtype='float32')
    del lineArray

Duration = datetime.now() - startTime
print("Duration of Reading Data == "+str(Duration))
startTime = datetime.now()

#Some outputs about the computing device
device = torch.device(args.deviceName)
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.current_device())
    print(torch.cuda.is_available())

#Convert numpy array to pytorch tensor
x = torch.from_numpy(data).float().to(device)
#Create tensor x which is normalize input data on each column
XMean = torch.mean(x,0).to(device)
XStd = torch.std(x,0).to(device)
x = (x-XMean.expand_as(x))/XStd.expand_as(x).to(device)
del XMean
del XStd

#Compute SVD decomposition of the normalized tensor x
u, s, v = torch.svd(x,some=True,compute_uv=True)
#and Project Data to new PCs
projectedData = torch.matmul(u,torch.diag(s)).to(device)
del u,s,v

#Apply Sign Flip for the projected data
if args.applySignFlip == 'yes':
    absProjectedData = torch.abs(projectedData).to(device)
    temp = torch.eq(absProjectedData,torch.max(absProjectedData,-2,keepdim=True).values).type(torch.FloatTensor).to(device)
    signMatrix = torch.sign(torch.sum(projectedData*temp,-2,keepdim=True).to(device)).to(device)
    projectedData = projectedData*signMatrix.to(device)
Duration = datetime.now() - startTime
print("Duration of Execution == "+str(Duration))
startTime = datetime.now()

#Output the results
np.savetxt (args.outputPath, projectedData.cpu().numpy(), delimiter=",")
Duration = datetime.now() - startTime
print("Duration of Writing Data == "+str(Duration))

#Output some useful information if the compute was performed on cuda 
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


