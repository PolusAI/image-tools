FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

COPY src/requirements.txt .

RUN mkdir -m 777 -p ${DATA_DIR}/{inputs,outputs} && \
    mkdir -m 777 -p ${EXEC_DIR}/numba_cache && \
    pip3 install -r requirements.txt --no-cache-dir

ENV NUMBA_CACHE_DIR=${EXEC_DIR}/numba_cache

# Change to .ome.zarr to save output images as zarr files.
ENV POLUS_EXT=".ome.tif"

# Change to WARNING for fewer logs, or DEBUG for debugging.
ENV POLUS_LOG="INFO"

COPY VERSION /
COPY src ${EXEC_DIR}/

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "vector_to_label.py"]