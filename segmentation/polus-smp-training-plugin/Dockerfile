FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

COPY requirements.txt .

RUN pip3 install -r requirements.txt --no-cache-dir && \
    mkdir -p ${EXEC_DIR} && \
    mkdir -p ${DATA_DIR}/{pretrained_model, images, labels, output} && \
    mkdir /.cache && chmod 777 /.cache

COPY VERSION /
COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
