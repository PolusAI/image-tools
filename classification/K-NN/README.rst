===================================
K-NN Code for Shared-Memory Systems
===================================

Please consider the following instruction for the execution of K-NN Code 
for Shared-Memory systems. The full description of the code is available 
`Here <https://labshare.atlassian.net/wiki/spaces/WIPP/pages/699039829/K-NN+Implementations+in+C+>`_.

-----------------
Runtime Arguments
-----------------

The code required 6 parameters as the input that are listed in order below.

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

------------------------------
An Example of Running the code
------------------------------

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
