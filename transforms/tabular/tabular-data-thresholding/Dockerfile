FROM python:3.9-slim

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

ENV FILE_EXT="csv"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN apt-get -y update && \
    apt-get -y install git && \
    pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir --force && \
    pip3 install git+https://github.com/nishaq503/thresholding.git

ENTRYPOINT ["python3", "main.py"]