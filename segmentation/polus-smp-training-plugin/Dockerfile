#FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

COPY requirements-docker.txt .

ENV CUDA_VISIBLE_DEVICES=0

RUN apt-get -y update && \
    apt-get -y install git && \
    pip3 install -r requirements-docker.txt --no-cache-dir && \
    mkdir -p ${EXEC_DIR} && \
    mkdir -p ${DATA_DIR}/{pretrained_model, images, labels, output} && \
    mkdir /.cache && chmod 777 /.cache

COPY VERSION /
COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "/opt/executables/main.py"]
