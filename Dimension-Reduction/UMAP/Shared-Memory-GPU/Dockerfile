FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Update apps on the base image
RUN apt-get -y update && apt-get install -y g++ wget make libarmadillo-dev
 
#Create new directory 
RUN mkdir -p /home/GPU_UMAP
 
# Specify the working directory
WORKDIR /home/GPU_UMAP
 
# Install Boost Library 
RUN wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
RUN tar xfz boost_1_71_0.tar.gz 
RUN rm boost_1_71_0.tar.gz
WORKDIR /home/GPU_UMAP/boost_1_71_0
RUN ./bootstrap.sh
RUN ./b2
ENV LD_LIBRARY_PATH="/home/GPU_UMAP/boost_1_71_0/stage/lib:${LD_LIBRARY_PATH}" 

WORKDIR /home/GPU_UMAP

# Download Eigen3 Library 
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
RUN tar xfz eigen-3.3.7.tar.gz 
RUN rm eigen-3.3.7.tar.gz

# Copy the current folder to the docker image
COPY . /home/GPU_UMAP

#RUN mkdir /home/GPU_UMAP/input
#RUN cp /home/GPU_UMAP/mnist_784.csv /home/GPU_UMAP/input/

# Compile the source file 
WORKDIR /home/GPU_UMAP
RUN nvcc -I/home/GPU_UMAP/boost_1_71_0 -I/home/GPU_UMAP/eigen-3.3.7 main.cu  highDComputes.cpp Initialization.cpp LMOptimization.cpp Metrics.cpp KNN_GPU_Code.cu -larmadillo -o Out.exe -L/home/GPU_UMAP/boost_1_71_0/stage/lib -lboost_iostreams -lboost_system -lboost_filesystem -arch=sm_75 -O2 --expt-relaxed-constexpr

# Run the output program from the previous step
ENTRYPOINT ["./Out.exe"]



