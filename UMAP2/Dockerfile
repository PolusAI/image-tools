FROM ubuntu:latest

COPY VERSION /

RUN apt-get -y update && \
    apt-get -y install g++ wget make libarmadillo-dev


#Create a new directory
RUN mkdir -p /home/UMAP

# Specify the working directory
WORKDIR /home/UMAP
 
# Install Boost Library 
RUN wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
RUN tar xfz boost_1_71_0.tar.gz 
RUN rm boost_1_71_0.tar.gz
WORKDIR /home/UMAP/boost_1_71_0
RUN ./bootstrap.sh
RUN ./b2
ENV LD_LIBRARY_PATH="/home/UMAP/boost_1_71_0/stage/lib:${LD_LIBRARY_PATH}" 

WORKDIR /home/UMAP

# Download Eigen3 Library 
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
RUN tar xfz eigen-3.3.7.tar.gz 
RUN rm eigen-3.3.7.tar.gz

# Copy the current folder which contains C++ source code to the Docker image
COPY . /home/UMAP

RUN g++ -I/home/UMAP/boost_1_71_0 -I/home/UMAP/eigen-3.3.7 main.cpp KNN_Serial_Code.cpp highDComputes.cpp Initialization.cpp LMOptimization.cpp Metrics.cpp SGD.cpp -o Out.exe -O2 -larmadillo -L/home/UMAP/boost_1_71_0/stage/lib -lboost_iostreams -lboost_system -lboost_filesystem -fopenmp

ENTRYPOINT ["./Out.exe"]

