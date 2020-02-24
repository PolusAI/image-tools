===================================
PCA Implementation
===================================

Please refer to `this link <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/690585601/PCA+Implementations+in+PyTorch>`_ for detailed theoretical background about PCA.
PCA has been implemented in PyTorch in two ways for Shared-Memory systems and
Distributed-Memory systems. The Shared-Memory implementation is an ideal solution 
for relatively small-sized dataset where the entire data can be fit into the memory. 
For larger dataset, Distributed-Memory implementation is recommended where the
dataset is divided among multiple machines and each machine will perform independent
computing on a subset of the dataset. 

------------------------------------
Shared-Memory Systems Implementation
------------------------------------

The code requires six input arguments as listed in order below.

1- ``readOption``: This parameter is either 'direct' or 'mapping'. 'direct' 
                   represents the case where the entire data is brought to memory
                   and 'mapping' represents the case where data cannot fit into
                   the memory and reading is performed through mapping. 
2- ``deviceName``: This parameter defines the compute device and is either 'cpu' or 'cuda:0'. 
3- ``applySignFlip``: If this parameter is set to 'yes' the sign of the projected data in PCs space is flipped.
4- ``computeStdev``:  If this parameter is set to 'yes' the post-compute analysis will be performed on the PC axes 
                       and the standard deviation of the projected data will be computed along PC axes (column 1) along with
                       the ratio of (standard deviation for each axes)/(sum of standard deviations for all PC axes)*100 in (column 2)
5- ``inputPath``: The full path to the input csv file which contains raw data.
                   In this file, the observations are stored in rows and the features 
                   in columns.
6- ``outputPath``: The full path to the csv file where the projected data in PCs space 
                   are saved. This argument is optional and the default path is the
                   current directory with the file name PCA_Projected_Data_Final.csv                
                   
The code produces the following outputs.

1- ``outputPath.csv``: The output file where the projected data in PCs space are saved. 
                       The name of this output csv file was inserted from the input argument and the default name is PCA_Projected_Data_Final.csv
2- ``PCs.csv``:        The rows of this file represents the PC directions
3- ``SingularValues.csv``:  The singular values which were derived from SVD decompisition  
4- ``Setting.txt``:    The logging file containing the error and informational messages. 
5- ``Stdev.csv``:      This file is produced only if the input argument of "computeStdev" is set to "yes". 
                       This file contains 2 columns: the standard deviation of the projected data computed along PC axes (column 1) 
                       and the ratio of (standard deviation for each axes)/(sum of standard deviations for all PC axes)*100 in (column 2)
                   
--------------------------------------------
Installing PyTorch for Shared-Memory Systems
--------------------------------------------
The first step is to install conda as shown below.

.. code:: bash

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod 755 Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    conda create --name PyTorch_Shared Python=3.7.3 flask
    conda activate PyTorch_Shared 
    
Next, PyTorch is installed from the source as follows.
                     
 .. code:: bash

    #Install Dependencies in Conda:                  
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing pandas dask                 
    conda install -c pytorch magma-cuda101
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}              
    python setup.py install >> output.txt 2>&1
    
Now, the python code can be executed as follows. Please note that inputPath should only contain a single csv file which is the main input data. 
 
 .. code:: bash  
 
    python PCA_SVD_SharedMemory.py --readOption direct --deviceName cpu --applySignFlip true --computeStdev true --inputPath . --outputPath . 

docker can be run as follows.   
 .. code:: bash    
 
   sudo docker run -v /path/to/Docker:/data/inputs -v /path/to/Docker:/data/outputs  dockerImageName  --readOption direct --deviceName cpu --applySignFlip true --computeStdev true --inputPath /data/inputs  --outputPath /data/outputs
        
-----------------------------------------
Distributed-Memory Systems Implementation
-----------------------------------------

The code requires three input arguments as listed in order below.

1- ``deviceName``: The name of computing device which is either 'cpu' or 'gpu'. For now, 
                   the code has been tested for 'cpu' using MPI communication.
2- ``inputPath`` : The full path to the input csv file which contains raw data.
                   In this file, the observations are stored in rows and the features 
                   in columns.
3- ``outputPath``: The full path to the csv file where the projected data in PCs space 
                   are saved. This argument is optional and the default path is the
                   current directory with the file name PCA_Projected_Data_Final.csv

Also, for launching PyTorch using mpirun, the number of processors should also be defined after flag "-np".
An example of exectuing the code is given below. In this example, 2 processors will run the code simultaneously. 

.. code:: bash
mpirun -np 2 python PCA_Cov_DistributedMemory.py cpu /Path/input.csv /Path/output.csv


The code produces the following outputs.

1- ``outputPath.csv``: The output file where the projected data in PCs space are saved. 
                       The name of this output csv file was inserted from the input argument and the default name is PCA_Projected_Data_Final.csv
2- ``Setting.txt``:    The logging file containing the error and informational messages.  
3- ``eigenValues.csv``:  The eigen values of covariance matrix ordered in ascending order.
4- ``eigenVectors.csv``:  The corresponding eigen vectors of covariance matrix .                       

-------------------------------------------------
Installing PyTorch for Distributed-Memory Systems
-------------------------------------------------
The first step is to install conda as shown below.

.. code:: bash

    wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    chmod 755 Anaconda3-2019.03-Linux-x86_64.sh
    ./Anaconda3-2019.03-Linux-x86_64.sh
    conda create --name PyTorch_Dist Python=3.7.3 flask
    conda activate PyTorch_Dist

Next, the MPI version of PyTorch is installed as follows.

.. code:: bash

    #Install Dependencies in Conda:
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing pandas git
    #Install PyTorch:
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    #Install openmpi and PyTorch:
    export USE_CUDA=0
    conda install -c conda-forge openmpi
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    python setup.py install >> output.txt 2>&1

Now, PyTorch can be launched on multiple distributed machines as follows.

.. code:: bash

    #Execute the Code on Single machine, multiple processes:
    mpirun -np 2 python PCA_Cov_DistributedMemory.py cpu /Path/input.csv /Path/output.csv

    #Execute the Code on Multiple machines, multiple processes:
    mpirun --hostfile nodes.txt --map-by node -np 2 python PCA_Cov_DistributedMemory.py cpu /Path/input.csv /Path/output.csv
    #The nodes.txt file is a simple text file where machines IP are listed on each line. 

For more information about the installing PyTorch for distributed machines, refer to the following links:
https://github.com/pytorch/pytorch#from-source
https://pytorch.org/tutorials/intermediate/dist_tuto.html











