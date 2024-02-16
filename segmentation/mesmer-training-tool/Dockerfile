# Get Linux
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

# Container log level
ENV POLUS_LOG="INFO"

WORKDIR ${EXEC_DIR}
COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}

# Instal Python
RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 python3.9-dev python3.9-distutils curl openjdk-8-jre build-essential -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN pip install --index-url https://test.pypi.org/simple/ filepattern==2.2.8
COPY src ${EXEC_DIR}/src
RUN pip3 install ${EXEC_DIR} --no-cache-dir
# Fix for cudnn error on RTX gpus
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENTRYPOINT ["python3", "-m", "polus.images.segmentation.mesmer_training"]
CMD ["--help"]
