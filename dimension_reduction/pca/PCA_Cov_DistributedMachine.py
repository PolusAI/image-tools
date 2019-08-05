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

#------------------Define Device---------------------------------
devicename = 'cpu' 
device = torch.device(devicename)
#print('Using device:', device)
#print(torch.cuda.current_device())
#print(torch.cuda.is_available())

def readinput(rank,size):
    Lines_Total_Tensor=torch.tensor(1) 
    file_fullpath='/home/maghrebim2/Work/PCA_Implementations/PyTorch_Dist/Working_Dir/10.csv'
    file_name=file_fullpath.split('.')[0].split('/')[-1] 
    if (rank==0):
        output = check_output(["wc", "-l", file_fullpath])
        Lines_Total=int(output.decode("utf-8").split(' ')[0])
        Lines_Total_Tensor=torch.tensor(Lines_Total)  
        lineperprocessor=math.floor(Lines_Total_Tensor/size)+size          
        cmd="split "+ "-dl "+ str(lineperprocessor) + " " + file_fullpath+ \
        " --additional-suffix=.csv " + file_name + "_"
        os.system(cmd)
    torch.distributed.barrier(async_op=False)    
    if (size<10):
        split_file_name=file_name + "_0"+str(rank)+".csv"        
    else: 
        split_file_name=file_name + "_"+str(rank)+".csv"   
    if (rank==0):
        df=pd.read_csv(split_file_name)
    else:
        df=pd.read_csv(split_file_name,header=None)       
    d = df.values
    data = np.float32(d)  #float64 better here
    del d  
    cmd2='rm '+split_file_name
    os.system(cmd2)   #rm 10_0*.csv
    return data

def Normalize(data):
    x = torch.from_numpy(data).float().to(device)
    x_sum = torch.sum(x,0).to(device)
    x_counts=torch.tensor(x.shape[0])
    torch.distributed.all_reduce(x_sum,op=dist.ReduceOp.SUM,async_op=False)
    torch.distributed.all_reduce(x_counts,op=dist.ReduceOp.SUM,async_op=False)
    x_mean=x_sum/x_counts
    x_squared=torch.sum(torch.pow((torch.sub(x,x_mean[None,:])),2),0)
    torch.distributed.all_reduce(x_squared,op=dist.ReduceOp.SUM,async_op=False)   
    x_std=torch.sqrt(x_squared/(x_counts-1))
    x_Normalized=torch.sub(x,x_mean[None,:])/x_std[None,:]
    del x
    return x_Normalized

def ComputePCA(x_Normalized):    
    n=x_Normalized.shape[1]
    coefficient=1/(n-1)
    Cov_Matrix=torch.matmul(torch.transpose(x_Normalized, 0, 1),x_Normalized).to(device)
    Cov_Matrix=torch.mul(Cov_Matrix,coefficient)
    torch.distributed.all_reduce(Cov_Matrix,op=dist.ReduceOp.SUM,async_op=False) 
    EigenValues, EigenVectors = torch.symeig(Cov_Matrix, eigenvectors=True)
    return EigenVectors 

def Project_Data(x_Normalized,EigenVectors,rank,size):
    projected_Data= torch.matmul(x_Normalized,EigenVectors).to(device)
    outputfile="/home/maghrebim2/Work/PCA_Implementations/PyTorch_Dist/Working_Dir/PCA_Projected_Data_"+str(rank)+".csv"
    np.savetxt(outputfile,projected_Data.cpu().numpy(), delimiter=",")  
    if (rank==0): 
        cmd='cat '
        for i in range(size):
            cmd+=" PCA_Projected_Data_"+str(i)+".csv "
        cmd+=(">  PCA_Projected_Data_Final.csv")
        os.system(cmd)
        for i in range(size):
            cmd2="rm PCA_Projected_Data_"+str(i)+".csv "      
            os.system(cmd2)  #rm PCA_Projected_Data_*.csv

def init_processes(rank, size, hostname, fn, backend='mpi'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    startTime = datetime.now()
    outputdata= readinput(rank,size)
    Duration = datetime.now() - startTime
    print("Rank #"+str(rank)+ "says, it took "+str(Duration)+ "for READING INPUT DATA") 
    startTime = datetime.now()
    output_Normalized=Normalize(outputdata)
    output_EigenVectors=ComputePCA(output_Normalized)
    Duration = datetime.now() - startTime
    print("Rank #"+str(rank)+ "says, it took "+str(Duration)+ "for COMPUTING PCA") 
    startTime = datetime.now()
    Project_Data(output_Normalized,output_EigenVectors,rank,size)
    Duration = datetime.now() - startTime
    print("Rank #"+str(rank)+ "says, it took "+str(Duration)+ "for Writing Outputs")    
    
if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    hostname = socket.gethostname()
    init_processes(world_rank, world_size, hostname, readinput, backend='mpi')

