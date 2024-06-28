FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"
ENV TF_FORCE_GPU_ALLOW_GROWTH=true


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

WORKDIR ${EXEC_DIR}

# TODO: Change the tool_dir to the tool directory
ENV TOOL_DIR="segmentation/mesmer-training-tool"

# Copy the repository into the container
RUN mkdir image-tools
COPY . ${EXEC_DIR}/image-tools

# Install the tool
RUN pip3 install "${EXEC_DIR}/image-tools/${TOOL_DIR}" --no-cache-dir

# Fix for cudnn error on RTX gpus
RUN python3 -c "import deepcell;from deepcell.applications import Mesmer; app = Mesmer()"

# Set the entrypoint
# TODO: Change the entrypoint to the tool entrypoint
ENTRYPOINT ["python3", "-m", "polus.images.segmentation.mesmer_training"]
CMD ["--help"]
