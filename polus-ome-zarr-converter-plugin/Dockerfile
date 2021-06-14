
FROM labshare/polus-bfio-util:2.1.8

# Directory where code should be placed
ARG EXEC_DIR="/opt/executables"

# Directories for mounting data to the container
ARG DATA_DIR="/data"
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]