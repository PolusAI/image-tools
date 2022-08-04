FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# environment variables defined in labshare/polus-bfio-util
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"
ENV POLUS_EXT=".ome.tif"

# Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

RUN cat /etc/os-release

# Install Python and Java
RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 python3.9-distutils curl openjdk-8-jre -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3.9", "main.py"]
