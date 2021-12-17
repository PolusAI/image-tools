FROM ktaletsk/gpu:bionic-20190612-cuda-10-2

# Update apps on the base image
RUN apt-get -y update && apt-get install -y g++ wget make
 
#Create new directory 
RUN mkdir -p /home/GPU_KNN /home/Inputs /home/Outputs
 
# Specify the working directory
WORKDIR /home/GPU_KNN
 
# Install Boost Library 
RUN wget https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
RUN tar xfz boost_1_71_0.tar.gz 
RUN rm boost_1_71_0.tar.gz
WORKDIR /home/GPU_KNN/boost_1_71_0
RUN ./bootstrap.sh
RUN ./b2
ENV LD_LIBRARY_PATH="/home/GPU_KNN/boost_1_71_0/stage/lib:${LD_LIBRARY_PATH}" 

# Copy the current folder to the docker image
COPY . /home/GPU_KNN

# Compile the source file 
WORKDIR /home/GPU_KNN
RUN nvcc -I/home/GPU_KNN/boost_1_71_0 KNN_GPU_Code.cu -o Out.exe -L/home/GPU_KNN/boost_1_71_0/stage/lib -lboost_iostreams -lboost_system -lboost_filesystem -arch=sm_75 -O2
 
# Run the output program from the previous step
ENTRYPOINT ["./Out.exe"]



