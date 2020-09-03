
FROM labshare/polus-bfio-util:2.4.4

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src/requirements.txt ${EXEC_DIR}/
RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

COPY src ${EXEC_DIR}/
COPY VERSION /
WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]