import numpy as np
import pandas as pd
from datetime import datetime
import torch
import mmap
import dask.dataframe as ddf
import dask.multiprocessing
import argparse
import logging
import os
import glob

argparser = argparse.ArgumentParser()
argparser.add_argument('--readOption', dest='readOption', type=str)
argparser.add_argument('--deviceName', dest='deviceName', type=str)
argparser.add_argument('--applySignFlip', dest='applySignFlip', type=str)
argparser.add_argument('--computeStdev', dest='computeStdev', type=str)
argparser.add_argument('--inputPath', dest='inputPath', type=str)
argparser.add_argument('--outputPath', dest='outputPath', type=str, nargs='?', default='./PCA_Projected_Data_Final.csv')
args = argparser.parse_args()

# Find the first CSV file in the input folder
inputPath = glob.glob(os.path.join(args.inputPath, "*.csv"))[0]
# Set the path to the output files
outputPath = os.path.join(args.outputPath, 'PCA_Projected_Data_Final.csv')
SingularValuesOutputPath = os.path.join(args.outputPath, 'SingularValues.csv')
PCsOutputPath = os.path.join(args.outputPath, 'PCs.csv')
StdevOutputPath = os.path.join(args.outputPath, 'Stdev.csv')
SettingOutputPath = os.path.join(args.outputPath, 'Setting.txt')

logging.basicConfig(filename=SettingOutputPath , level=logging.INFO)

startTime = datetime.now()
#Reading input data directly and creating a numpy array.
if args.readOption=='direct':
    df = ddf.read_csv(inputPath,sep=',')   
    d = df.compute(scheduler='threads') 
    data = np.float32(d)  
    del d

#Mapping Data to Memory and create a numpy array.
#This method is good if data cannot be fit into the memory.
elif args.readOption == 'mapping':
    fileName = open(inputPath, "r")
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

duration = datetime.now() - startTime
logging.info("Duration of Reading Data == "+str(duration))
startTime = datetime.now()

#Some outputs about the computing device
device = torch.device(args.deviceName)
logging.info("Using device:"+str(device))
if device.type == 'cuda':
    logging.info('torch.cuda.current_device()= '+ str(torch.cuda.current_device()))
    logging.info('torch.cuda.is_available()= '+ str(torch.cuda.is_available()))

#Convert numpy array to pytorch tensor
x = torch.from_numpy(data).float().to(device)
featureCounts=np.shape(data)[-1]
#del data

#Create tensor x which is normalize input data on each column
XMean = torch.mean(x,0).to(device)
XStd = torch.std(x,0).to(device)
x = (x-XMean.expand_as(x))/XStd.expand_as(x).to(device)
del XMean
del XStd

#Compute SVD decomposition of the normalized tensor x
#PyTorch outputs v Matrix and not v.t()
u, s, v = torch.svd(x,some=True,compute_uv=True)

np.savetxt (SingularValuesOutputPath, s.cpu().numpy(), delimiter=",", header="Singluar Values", comments='')

strs = ["Axis" for x in range(featureCounts)]
nums=list(range(1,featureCounts+1))
headerLiterals=''.join(n+str(s)+',' for (n,s) in zip(strs, nums))
np.savetxt (PCsOutputPath, v.t().cpu().numpy(), delimiter=",", header=headerLiterals, comments='')
#and Project Data to new PCs
projectedData = torch.matmul(u,torch.diag(s)).to(device)

#Compute the Standard Deviation of the projected data along each PC axis
if args.computeStdev == 'true':
    Stdev = torch.std(projectedData,0).to(device)
    SumStdev = torch.sum(Stdev).to(device)
    normalizedStdev= torch.mul(torch.div(Stdev,SumStdev).to(device),100).to(device)
    headerLiteral="Standard Deviation of Data Along each PC, Normalized Value in Percent" 
    np.savetxt (StdevOutputPath, torch.stack((Stdev,normalizedStdev),1).cpu().numpy(), delimiter=",", header=headerLiteral, comments='')
    del Stdev,SumStdev,normalizedStdev
del u,s,v

#Apply Sign Flip for the projected data
if args.applySignFlip == 'true':
    absProjectedData = torch.abs(projectedData).to(device)
    temp = torch.eq(absProjectedData,torch.max(absProjectedData,-2,keepdim=True).values).type(torch.FloatTensor).to(device)
    signMatrix = torch.sign(torch.sum(projectedData*temp,-2,keepdim=True).to(device)).to(device)
    projectedData = projectedData*signMatrix.to(device)
duration = datetime.now() - startTime
logging.info("Duration of Execution == "+str(duration))
startTime = datetime.now()

#Output the results
strs = ["PC" for x in range(featureCounts)]
nums=list(range(1,featureCounts+1))
headerLiterals=''.join(n+str(s)+',' for (n,s) in zip(strs, nums))
np.savetxt (outputPath, projectedData.cpu().numpy(), delimiter=",", header=headerLiterals, comments='')

duration = datetime.now() - startTime
logging.info("Duration of Writing Data == "+str(duration))

#Output some useful information if the compute was performed on cuda 
if device.type == 'cuda':
    logging.info(torch.cuda.get_device_name(0))
    logging.info('Memory Usage:')
    logging.info('Allocated:'+ str(round(torch.cuda.memory_allocated(0)/1024**3,1))+ 'GB')
    logging.info('Cached:   '+ str(round(torch.cuda.memory_cached(0)/1024**3,1))+ 'GB')


