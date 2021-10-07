FROM python:3.7 

ENV POLUS_LOG="DEBUG"
ENV POLUS_EXT=".ome.tif"

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

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt --no-cache-dir

RUN apt update \
    && apt-get install -y git \
    && apt-get install -y libgl1-mesa-dev

RUN git clone --branch training_big https://github.com/mmvih/splinedist.git
RUN pip3 install cmake --no-deps
RUN pip3 install splinedist/ 

WORKDIR ${EXEC_DIR}
ENTRYPOINT ["python3", "main.py"]

