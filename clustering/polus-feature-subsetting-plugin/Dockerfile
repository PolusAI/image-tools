
FROM polusai/bfio:2.1.9

# from bfio container
# ENV POLUS_EXT=".ome.tif"
# ENV POLUS_LOG="INFO"
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "/opt/executables/main.py"]