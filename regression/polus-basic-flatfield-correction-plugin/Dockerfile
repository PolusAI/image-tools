FROM python:3.9-slim
LABEL maintainer="Axle Informatics"

COPY VERSION /

ENV DEBIAN_FRONTEND noninteractive
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]