FROM polusai/bfio:2.1.9

# Change to .ome.zarr to save output images as zarr files.
ENV POLUS_EXT=".ome.tif"

# Change to WARNING for fewer logs, or DEBUG for debugging.
ENV POLUS_LOG="INFO"

ARG DATA_DIR="/data"
ARG EXEC_DIR="/opt/executables"

COPY requirements.txt /

RUN pip3 install -r requirements.txt --no-cache-dir && \
    mkdir -p ${DATA_DIR}/{inputs, outputs} && \
    mkdir -p ${EXEC_DIR}

COPY VERSION /
COPY src ${EXEC_DIR}/

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "/opt/executables/main.py"]