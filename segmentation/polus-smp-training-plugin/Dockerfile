FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

COPY requirements-docker.txt .

RUN apt-get -y update && \
    apt-get -y install git && \
    pip3 install -r requirements-docker.txt --no-cache-dir && \
    pip3 install git+https://github.com/PolusAI/segmentation_models.pytorch.git && \
    mkdir -p ${EXEC_DIR} && \
    mkdir -p ${DATA_DIR}/{pretrained_model, images, labels, output} && \
    mkdir /.cache && chmod 777 /.cache

COPY VERSION /
COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
