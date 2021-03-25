FROM python:3.6-slim

COPY VERSION /
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir
WORKDIR ${EXEC_DIR}
ENTRYPOINT ["python3", "main.py"]

FROM python:3.6-slim
COPY VERSION /
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN apt update
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y build-essential

RUN git clone --branch decomposition https://github.com/mmvih/neurogen.git --recurse-submodules
RUN pip3 install cmake
RUN pip3 install neurogen/



