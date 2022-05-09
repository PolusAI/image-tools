FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} && mkdir -p ${DATA_DIR} && \
    pip3 install "bfio[all]"==2.1.9

COPY requirements-docker.txt .
RUN pip3 install -r requirements-docker.txt --no-cache-dir

COPY VERSION /
COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
