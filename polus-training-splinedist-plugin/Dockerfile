FROM python:3.7-slim as python

COPY VERSION /
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs
COPY src ${EXEC_DIR}/
RUN mkdir -p ${EXEC_DIR}
COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir
WORKDIR ${EXEC_DIR}
ENTRYPOINT ["python3", "main.py"]

FROM python as splinedist

RUN git clone --branch training https://github.com/mmvih/splinedist.git

