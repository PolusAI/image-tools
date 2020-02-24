===================================
K-NN Code for Shared-Memory Systems
===================================

Please consider the following instruction for the execution of K-NN Code 
for Shared-Memory systems. The full description of the code is available 
`Here <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/699039829/K-NN+Implementations+in+C+>`_.

------------------------
Installing Boost Library
------------------------

Both K-NN codes for Shared-Memory and Distributed-Memory use Boost library for mapping data into memory and reading from the command line. The steps for installing Boost library are displayed below.
 
.. code:: bash
    
    wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
    tar xfz boost_1_71_0.tar.gz 
    cd boost_1_71_0/
    ./bootstrap.sh
    ./b2
    export LD_LIBRARY_PATH=currentpath/stage/lib:$LD_LIBRARY_PATH

It is recommended to include the last line in the above into .bashrc file at home directory. 

-----------------
Runtime Arguments
-----------------

The code requires the following parameters as the input that are listed in order below.

1- ``filePath``: The full path to the input csv file containig the dataset. Please ensure there are no
                   other csv files in this path. 
2- ``K``: The desired number of Nearest Neighbours to be computed.
3- ``sampleRate``: The rate at which we do sampling. This parameter plays a key role
   in the performance. This parameter is a trades-off between the performance 
   and the accuracy of the results. Values closer to 1 provides more accurate
   results but the execution instead takes longer.    
4- ``convThreshold``: An integer that controls the convergence of the model. A fixed
   integer is used here instead of delta*N*K that was given in the paper. 
5- ``outputPath``: The full path to the output csv files.    
6,7- ``colIndex1`` and ``colIndex2`` (Optional): The indices of columns from the input csv file where raw data exists continuously in between. If these two arguments were left blank, the code assumes that the entire input csv file is raw data and automatically computes the number of columns in the input csv file. The numbering for these 2 indices begin from 1. Please note that the code assumes that the first line in the input csv file is the header.

-----------
The Outputs
-----------

The code produces the following output files:

1- ``KNN_Indices.csv``: The indices of K-NNs for the entire dataset. The order of data here is the same as the order of data at the input csv file.
2- ``KNN_Distances.csv``: The corresponding distances of K-NNs which was saved at KNN_Indices.csv.
3- ``Setting.txt``: The logging file containing the error and informational messages. 

--------------------------------
An Example of Executing the Code
--------------------------------

.. code:: bash

    ulimit -s unlimited
    g++ -I/Path_To_Boost_Library/boost_1_71_0 KNN_Serial_Code.cpp -o a.out -L/Path_To_Boost_Library/boost_1_71_0/stage/lib -lboost_iostreams -lboost_system -lboost_filesystem  -O2 
    time ./a.out --inputPath . --K 10 --sampleRate 0.99  --convThreshold 5  --outputPath .
    time ./a.out --inputPath . --K 10 --sampleRate 0.99  --convThreshold 5  --outputPath .  --colIndex1 3 --colIndex2 26
    
---------------------------
An Advise About Performance
---------------------------

The parameter sampleRate has a significant impact on the performance. It is advised that its optimal value to be determined for every specific project. 

-------------------
Install WIPP Plugin
------------------- 
If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of plugin.json into the pop-up window and submit.
   
------------------------------------------
An Example of Running the Docker Container
------------------------------------------  

.. code:: bash

    docker run -v /path/to/data:/data/inputs -v /path/to/outputs:/data/outputs \
            containername --inputPath /data/inputs --K 10 --sampleRate 0.9 \
            --convThreshold 5 --outputPath /data/outputs          

========================================
K-NN Code for Distributed-Memory Systems
========================================

Please consider the following instruction for the execution of K-NN Code 
for Distributed-Memory systems. The full description of the code is available 
`Here <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/699039829/K-NN+Implementations+in+C+>`_.

-----------------
Runtime Arguments
-----------------

The code requires the following input parameters that are listed in the order.

1- ``Number of Processors``: Due to the design of global Kd Tree, the number of processors should be a power of 2 (1,2,4,8,16,...). 
2- ``filePath``: The full path to the input csv file containig the raw dataset.
3- ``KNNCounts``: The desired number of Nearest Neighbours to be computed.
4- ``colIndex1`` and ``colIndex2`` (Optional): The index of columns from the input csv file where raw data exists continuously in between. If these two arguments were left blank, the code assumes that the entire input csv file is raw data and automatically computes the number of columns in the input csv file. The numbering for these 2 indices begin from 1. Please note that the code assumes that the first line in the input csv file is the header.

Also, the execution performance has been improved by using OpenMP directives (multi-threading) in addition to the current MPI directives (multi-node). The number of threads in the OpenMP parallelized region of the code is set using an environment variable as shown below: 

.. code:: bash
    export OMP_NUM_THREADS=2

--------------------------------
An Example of Executing the code
--------------------------------

.. code:: bash

    ulimit -s unlimited
    export OMP_NUM_THREADS=2
    mpicxx -I/Path_To_Boost_Library/boost_1_71_0 KNN_Distributed_code-OpenMP.cpp -o output.exe -L/Path_To_Boost_Library/boost_1_71_0/stage/lib -lboost_iostreams -O2 -fopenmp
    time mpirun -np 4 ./output.exe /fullPath/inputfile.csv 15
    time mpirun -np 4 ./output.exe /fullPath/inputfile.csv 15 3 26
    
-----------
The Outputs
-----------

Similar to the shared memory KNN code, the distributed memory code produces the following output files:

1- ``KNN_Indices.csv``: The indices of K-NNs for the entire dataset. The first entry of each row contains the index of that point according to the index from the input csv file.
2- ``KNN_Distances.csv``: The corresponding distances of K-NNs which was saved at KNN_Indices.csv. The first entry of each row contains the index of that point according to the index from the input csv file.
3- ``Setting.txt``: The logging file containing the error and informational messages. 
   
---------------------------------------------------------
Description of the Other Important Parameters of the Code
---------------------------------------------------------

The code also has a few other parameters (listed below) that are a a part of the Kd Tree design. 
These parameters were initialized to the values suggested in the referencing paper (Patwary et al., 2016). 
For the complicated cases, these values might need to be changed for the optimized performance.     
 
1- ``globalKdTreeSamples``: The number of data sampled by each processor to collaboratively compute dimensions with the highest variability.
2- ``globalKdTreeSamplesMedian``: The number of data sampled by each processor to collaboratively compute the median of the chosen dimension for each splitting node within the global Kd Tree.
3- ``Parallel_IO``: A flag that defines if the input csv file can be read in parallel by all the processors. 
4-``Epsilon``: A buffer in accepting the Median value.
5- ``localKdTreeSamplesMedian``: The number of data sampled by each processor separately to compute the median of the chosen dimension for each splitting node within the local Kd Tree.
6- ``bucketSize``: The size of a bucket (or a leaf) in the local Kd Tree.
7- ``estimatedExtraLayers``: To limit the growing size of the local Kd Trees, the growth of the tree is limited by a number of layers defined here from the initial guess of the required buckets.
 
