'''
 * @author      Mahdi Maghrebi <mahdi.maghrebi@nih.gov>
 * July 2019
'''
import os
import socket
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
from subprocess import check_output
import math
import argparse
import logging

'''
readInput reads the input file and returns it as a numpy 2D array
@param rank the rank of each processor
@param size total number of processors
@return input data as a numpy 2D array
'''
def readInput(rank,size):
    linesTotalTensor = torch.tensor(1) 
    fileName = args.inputPath.split('.')[0].split('/')[-1] 
    #Procssor 0 splits the input data and creates one input file per processor
    if (rank == 0):
        output = check_output(["wc", "-l", args.inputPath])
        linesTotal = int(output.decode("utf-8").split(' ')[0])
        linesTotalTensor = torch.tensor(linesTotal)  
        linePerprocessor = math.floor(linesTotalTensor/size)+size          
        cmd = "split "+ "-dl "+ str(linePerprocessor) + " " + args.inputPath+ \
        " --additional-suffix=.csv " + fileName + "_"
        os.system(cmd)
    #Halt all the processors until procssor 0 returns    
    torch.distributed.barrier(async_op=False)  
    #Each processor reads its own data and converts it to a numpy array
    splitFileName = "{}_{:02d}.csv".format(fileName, rank)
    if (rank == 0):
        df=pd.read_csv(splitFileName)
    else:
        df = pd.read_csv(splitFileName,header=None)       
    d = df.values
    data = np.float32(d)  
    del d  
    #remove splitted input file for each processor
    cmd2 = 'rm ' + splitFileName
    os.system(cmd2)
    return data
    
'''
Normalize the input data using Z-score method 
@param data the input data as a 2D numpy array
@return A PyTorch tensor with data normalized in each column
'''
def Normalize(data):
    #convert numpy array to pytorch tensor
    x = torch.from_numpy(data).float().to(args.deviceName)
    #compute the sum of each column for local data
    xSum = torch.sum(x,0).to(args.deviceName)
    #compute the count of rows for local data
    xCounts=torch.tensor(x.shape[0])
    #compute the sum of each column for the entire global data
    torch.distributed.all_reduce(xSum,op=dist.ReduceOp.SUM,async_op=False)
    #compute the count of rows for the entire global data
    torch.distributed.all_reduce(xCounts,op=dist.ReduceOp.SUM,async_op=False)
    #compute the Mean of each column for the entire global data 
    xMean=xSum/xCounts
    #compute (x-xmean)^2 for local data
    xSquared=torch.sum(torch.pow((torch.sub(x,xMean[None,:])),2),0)
    #compute (x-xmean)^2 for the entire global data  
    torch.distributed.all_reduce(xSquared,op=dist.ReduceOp.SUM,async_op=False) 
    #compute Standard Deviation for local data of each column  
    xStd=torch.sqrt(xSquared/(xCounts-1))
    #compute Z-Score Normalization for local data 
    xNormalized=torch.sub(x,xMean[None,:])/xStd[None,:]
    #delete pytorch tensor for the local data 
    del x,xSum,xMean,xStd,xSquared,xCounts
    #return tensor of normalized data
    return xNormalized
    
'''
Create Covariance Matrix and compute its eigenVectors
@param xNormalized PyTorch tensor with data normalized in each column
@return eigenVectors of Covariance Matrix
'''
def ComputePCA(xNormalized):    
    n = xNormalized.shape[1]
    coefficient = 1/(n-1)
    covMatrix = torch.matmul(torch.transpose(xNormalized, 0, 1),xNormalized).to(args.deviceName)
    covMatrix = torch.mul(covMatrix,coefficient)
    torch.distributed.all_reduce(covMatrix,op=dist.ReduceOp.SUM,async_op=False) 
    eigenValues, eigenVectors = torch.symeig(covMatrix, eigenvectors=True)
    np.savetxt("eigenValues.csv",eigenValues.cpu().numpy(), delimiter=",") 
    np.savetxt("eigenVectors.csv",eigenVectors.cpu().numpy(), delimiter=",") 
    return eigenVectors 
    
'''
Project data to PCs space by multiplying normalized data with Eigenvectors
The results are saved in outputPath (second input argument from command line)
and the temporary files are removed
@param xNormalized PyTorch tensor with data normalized in each column
@param eigenVectors eigenvectors of Covariance Matrix
@param rank the rank of each processor
@param size total number of processors
'''
def Project_Data(xNormalized,eigenVectors,rank,size):
    projectedData = torch.matmul(xNormalized,eigenVectors).to(args.deviceName)
    outputFile = "tmpData_"+str(rank)+".csv"
    np.savetxt(outputFile,projectedData.cpu().numpy(), delimiter=",")  
    if (rank == 0): 
        cmd = 'cat '
        for i in range(size):
            cmd += " tmpData_"+str(i)+".csv "
        cmd += ("> "+ str(args.outputPath))
        os.system(cmd)
        #rm tmpData_*.csv
        for i in range(size):
            cmd2 = "rm tmpData_"+str(i)+".csv "      
            os.system(cmd2)  
            
'''
Initialize processors and calling different funcions
@param rank the rank of each processor
@param size total number of processors
@param backend the communication method between processors 
'''
def init_processes(rank, size, backend='mpi'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    startTime = datetime.now()
    TmpOutputData = readInput(rank,size)
    Duration = datetime.now() - startTime
    logging.info("Rank # "+str(rank)+ " says, it took "+str(Duration) + " for READING INPUT FILE") 
    print("Rank # "+str(rank)+ " says, it took "+str(Duration) + " for READING INPUT FILE") 
    startTime = datetime.now()
    TmpOutputNormalized = Normalize(TmpOutputData)
    TmpOutputEigenVectors = ComputePCA(TmpOutputNormalized)
    Duration = datetime.now() - startTime
    logging.info("Rank # "+str(rank)+ " says, it took "+str(Duration)+ " for COMPUTING PCA") 
    print("Rank # "+str(rank)+ " says, it took "+str(Duration)+ " for COMPUTING PCA") 
    startTime = datetime.now()
    Project_Data(TmpOutputNormalized,TmpOutputEigenVectors,rank,size)
    Duration = datetime.now() - startTime
    logging.info("Rank # "+str(rank) + " says, it took "+str(Duration)+ " for WRITING OUTPUTS")    
    print("Rank # "+str(rank) + " says, it took "+str(Duration)+ " for WRITING OUTPUTS")  
'''
Code begins from here 
''' 
if __name__ == "__main__":
    logging.basicConfig(filename="Setting.txt", level=logging.INFO)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('deviceName', type=str)
    argparser.add_argument('inputPath', type=str)
    argparser.add_argument('outputPath', type=str, nargs='?', default='./PCA_Projected_Data_Final.csv')
    args = argparser.parse_args()
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    init_processes(world_rank, world_size, 'mpi')

