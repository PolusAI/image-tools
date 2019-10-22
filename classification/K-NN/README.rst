===================================
K-NN Code for Shared-Memory Systems
===================================

Please consider the following instruction for the execution of K-NN Code 
for Shared-Memory systems. The full description of the code is available 
`Here <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/699039829/K-NN+Implementations+in+C+>`_.

-----------------
Runtime Arguments
-----------------

The code requires 6 parameters as the input that are listed in order below.

1- ``filePath``: The full path to the input csv file containig the dataset.
2- ``N``: Size of input dataset without the header (i.e.(#Rows in input dataset)-1).
3- ``Dim``: Dimension of input dataset (#Columns)
4- ``K``: the desired number of Nearest Neighbours to be computed.
5- ``sampleRate``: the rate at which we do sampling. This parameter plays a key role
   in the performance. This parameter is a trades-off between the performance 
   and the accuracy of the results. Values closer to 1 provides more accurate
   results but the execution instead takes longer.    
6- ``convThreshold``: An integer that controls the convergence of the model. A fixed
   integer is used here instead of delta*N*K that was given in the paper.  

--------------------------------
An Example of Executing the code
--------------------------------

.. code:: bash

    ulimit -s unlimited
    g++ -O2 KNN_Serial_Code.cpp -o a.out
    time ./a.out /home/K-NN_Implementation/Dataset.csv 37615 33 15 0.8 5

---------------------------
An Advise About Performance
---------------------------
2 parameters of sampleRate and largestDistance have significant impact on 
the performance. It is advised that their exact values to be determined for
every project. If the largest possible distance between pairs of datapoints
is much smaller than the pre-set value of largestDistance, it is highly 
recommended to update this parameter with the largest possible distance. 


========================================
K-NN Code for Distributed-Memory Systems
========================================

Please consider the following instruction for the execution of K-NN Code 
for Distributed-Memory systems. The full description of the code is available 
`Here <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/699039829/K-NN+Implementations+in+C+>`_.

-----------------
Runtime Arguments
-----------------

The code requires 4 input parameters that are listed in the order below.

1- ``Number of Processors``: Due to the design of global Kd Tree, the number of processors should be a power of 2 (1,2,4,8,16,...). 
2- ``filePath``: The full path to the input csv file containig the raw dataset.
3- ``featureCounts``: The dimension of the input dataset (#Columns in the filePath)
4- ``KNNCounts``: The desired number of Nearest Neighbours to be computed.

Also, the execution performance has been improved by using OpenMP directives (multi-threading) in addition to the current MPI directives (multi-node). The number of threads in the OpenMP parallelized region of the code is set using an environment variable as shown below: 

.. code:: bash
    export OMP_NUM_THREADS=2

------------------------
Installing Boost Library
------------------------

The code uses Boost library for mapping data into memory. The steps for installing
 Boost library are displayed below.
 
.. code:: bash
    
    wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
    tar xfz boost_1_71_0.tar.gz 
    cd boost_1_71_0/
    ./bootstrap.sh
    ./b2
    export LD_LIBRARY_PATH=currentpath/stage/lib:$LD_LIBRARY_PATH

It is recommended to include the last line in the above into .bashrc file at home directory. 

--------------------------------
An Example of Executing the code
--------------------------------

.. code:: bash

    ulimit -s unlimited
    export OMP_NUM_THREADS=2
    mpicxx -I/Path_To_Boost_Library/boost_1_71_0 KNN_Distributed_code-OpenMP.cpp -o output.exe -L/Path_To_Boost_Library/boost_1_71_0/stage/lib -lboost_iostreams -O2 -fopenmp
    time mpirun -np 4 ./output.exe /fullPath/inputfile.csv 165 9
    
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
 
