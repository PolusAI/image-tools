===================================
UMAP Code for Shared-Memory Systems
===================================

Please consider the following instruction for the execution of UMAP code for Shared-Memory systems. Please refer to `this link <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/745537586/UMAP+Implementations+in+C+>`_ for detailed theoretical background about UMAP.

-------------------------------
Installing the Required Library
-------------------------------

UMAP requires three external libraries of Boost, Armadillo, and Eigen3 for the execution. 
The steps for installing Boost library are explained below.
 
.. code:: bash
    
    wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
    tar xfz boost_1_71_0.tar.gz 
    cd boost_1_71_0/
    ./bootstrap.sh
    ./b2
    export LD_LIBRARY_PATH=currentpath/stage/lib:$LD_LIBRARY_PATH

It is recommended to include the above last line into ~/.bashrc file. 

The Armadillo library can be installed using the following command.

.. code:: bash

    sudo apt-get -y install libarmadillo-dev

The Eigen3 library is an header-only library and can be downloaded using the following command.

.. code:: bash

    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    tar xfz eigen-3.3.7.tar.gz 
    rm eigen-3.3.7.tar.gz
 
-----------------
Runtime Arguments
-----------------

The code required the following parameters as the input.

1- ``inputPath``: The full path to the directory that contains the input csv file. Please note that the code reads the first csv file in this directory.
                  Therefore, any other csv files (than the input csv file) should be deleted from this directory before running the program including the output from the previous run. 
 
2- ``K``: the desired number of Nearest Neighbours to be computed. If larger K was selected, UMAP better preserves the global distribution of data. 
          For smaller K values, UMAP instead preserves the local distribution of data. 

3- ``sampleRate``: the rate at which we do sampling in K-NN algorithm. This parameter plays a key role
                   in the performance. This parameter is a trades-off between the performance
                   and the accuracy of the K-NN estimates. The values closer to 1 provides more accurate
                   results but the execution instead takes longer. 
                      
4- ``DimLowSpace``: Dimension of Low-D (or embedding) space (Usually is between 1 to 3).

5- ``randomInitializing``: If set to true, the positions of data in the lower dimension space are initialized randomly; 
                           and if set to false, the positions are defined by solving Laplacian matrix using Armadillo library.  
                           
6- ``outputPath``: The full path to the directory in which the output files will be saved. 

7- ``n_epochs``: The total number of training epochs over the pairs of data points during SGD solution. 

8- ``min_dist``: defines how tight the points are from each other in Low-D space. The default value should be 0.001.

9- ``distanceMetric``: is the metric to compute the distance between the points in high-D space. The default value should be euclidean.

10- ``distanceV1``: is the first optional variable needed for computing distance in some metrics.

11- ``distanceV2``: is the second optional variable needed for computing distance in some metric.

12- ``inputPathOptionalArray``: The full path to the directory that contains a csv file of the optional array needed for computing distance in some metrics. 

-----------
The Outputs
-----------

The code produces the following output files at outputPath:

1- ``ProjectedData_EmbeddedSpace.csv``: The coordinates of the projected input data in the lower dimension space.

2- ``Setting.txt``: The logging file containing the error and informational messages. 

------------------------------
An Example of Running the code
------------------------------

.. code:: bash

    ulimit -s unlimited
    
    g++ -I/path to boost directory/boost_1_71_0 -I/Path to eigen3 directory/eigen-3.3.7 \
         main.cpp KNN_Serial_Code.cpp highDComputes.cpp Initialization.cpp LMOptimization.cpp \
         Metrics.cpp SGD.cpp -o a.out -O2 - armadillo \
         -L/path to boost directory/boost_1_71_0/stage/lib -lboost_iostreams \
         -lboost_system -lboost_filesystem -fopenmp
    
    time ./a.out --inputPath . --K 15 --sampleRate 0.8 --DimLowSpace 2 \
         --randomInitializing true --outputPath . --n_epochs 500 --min_dist 0.001 \
         --distanceMetric euclidean
    
-------------------
Install WIPP Plugin
------------------- 
If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of plugin.json into the pop-up window and submit.
   
------------------------------------------
An Example of Running the Docker Container
------------------------------------------  

.. code:: bash

    docker run -v /path/to/data:/data/inputs -v /path/to/outputs:/data/outputs \
          containername --inputPath /data/inputs --K 15 --sampleRate 0.8 \
          --DimLowSpace 2 --randomInitializing true --outputPath /data/outputs \
          --n_epochs 500 --min_dist 0.001 --distanceMetric euclidean
          
===========================
CUDA Implementation of UMAP
===========================
   
The inputs and outputs to CUDA UMAP code are the same as the serial code explained before. The state-of-the-art CUDA implementation of UMAP has significantly improved the performance of UMAP code. Two hotspots in UMAP execution were parallelized using CUDA directives which are computing KNN and solving for SGD solution. For KNN part, the number of thread blocks are equal to the number of datapoints and the distance computations for each pair of points is performed by each thread of the block. The variable MAXTPB controls the hardward limitation on the number of threads per block, which is by default 1024. If the computations within a block requires more threads, the computations will be carried on CPU instead. Furthermore, if the number of threads per block is very low, the computations will be on CPU too due to the performance benefits. This latter is controlled by the variable MinimumThreads. For SGD part, the number of threads per block is controlled by the variable TPB, which is by default 32. 

 
------------------------------
An Example of Running the code
------------------------------

.. code:: bash

    ulimit -s unlimited
    
    nvcc -I/path to boost directory/boost_1_71_0 -I/Path to eigen3 directory/eigen-3.3.7 \
         main.cu highDComputes.cpp Initialization.cpp LMOptimization.cpp Metrics.cpp KNN_GPU_Code.cu \
         -o a.out -larmadillo -L/path to boost directory/boost_1_71_0/stage/lib -lboost_iostreams \
         -lboost_system -lboost_filesystem -arch=sm_75 -O2 --expt-relaxed-constexpr -Xcompiler -fopenmp 
    
    time ./a.out --inputPath . --K 15 --sampleRate 0.8 --DimLowSpace 2 \
         --randomInitializing true --outputPath . --n_epochs 500 --min_dist 0.001 \
         --distanceMetric euclidean         
         
------------------------------------------
An Example of Running the Docker Container
------------------------------------------  

.. code:: bash

    docker run --gpus all -v /path/to/data:/data/inputs -v /path/to/outputs:/data/outputs \
          containername --inputPath /data/inputs --K 15 --sampleRate 0.8 \
          --DimLowSpace 2 --randomInitializing true --outputPath /data/outputs \
          --n_epochs 500 --min_dist 0.001 --distanceMetric euclidean

