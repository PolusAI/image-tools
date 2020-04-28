### 1. Get Linux
FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Install open JDK 8
RUN apt-get update && \
    apt-get -y install openjdk-8-jdk

RUN apt-get update && \    
    pip3 install --upgrade pip --no-cache-dir

# Create temporary folder
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}

# Copy bfio
COPY . ${EXEC_DIR}/

# Install bfio and requirements
RUN apt-get update && \        
    pip3 install --upgrade cython && \
    pip3 install -r  ${EXEC_DIR}/requirements.txt --no-cache-dir && \
    pip3 install ${EXEC_DIR}/ --no-cache-dir && \
    mv ${EXEC_DIR}/bfio/jars/loci_tools.jar /usr/local/lib/python3.6/dist-packages/bioformats/jars/loci_tools.jar -f && \
    rm -rf ${EXEC_DIR}

ARG DATA_DIR="/data"

WORKDIR ${EXEC_DIR}
