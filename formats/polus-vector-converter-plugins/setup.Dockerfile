FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements file
COPY src/requirements.txt .

# Set the cuda path for cupy
ENV CUDA_PATH=/usr/local/cuda/bin

# Instal Python
RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 python3.9-distutils curl openjdk-8-jre libgomp1 -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip3.9 install -r requirements.txt

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

# Change to .ome.zarr to save output images as zarr files.
ENV POLUS_EXT=".ome.tif"

# Change to WARNING for fewer logs, or DEBUG for debugging.
ENV POLUS_LOG="INFO"

RUN mkdir -p ${EXEC_DIR}
COPY src ${EXEC_DIR}/.

RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    rm /usr/local/bin/pip3 && \
    ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip3

COPY VERSION /
COPY src ${EXEC_DIR}/

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

RUN mkdir /.cupy && chmod 777 /.cupy

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "label_to_vector.py"]