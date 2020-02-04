### 1. Get Linux
FROM openjdk:8-slim-buster

# COPY VERSION /

# Install python
RUN apt-get update && \
    apt-get install -y python3.7 ca-certificates python3-pip python3.7-dev wget && \
    pip3 install --upgrade pip --no-cache-dir

# Create temporary folder
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}

# Copy bfio
COPY . ${EXEC_DIR}/

# Install bfio and requirements
RUN apt-get update && \
    apt-get install -y python3-dev wget && \
    pip3 install --upgrade cython && \
    pip3 install -r  ${EXEC_DIR}/requirements.txt --no-cache-dir && \
    pip3 install ${EXEC_DIR}/ --no-cache-dir && \
    mv ${EXEC_DIR}/bfio/jars/loci_tools.jar /usr/local/lib/python3.7/dist-packages/bioformats/jars/loci_tools.jar -f && \
    rm -rf ${EXEC_DIR}
		
ARG DATA_DIR="/data"

WORKDIR ${EXEC_DIR}