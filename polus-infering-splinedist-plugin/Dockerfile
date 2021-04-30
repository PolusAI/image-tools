FROM python:3.7 

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

RUN python3 -m pip install -r requirements.txt --no-cache-dir --ignore-installed

RUN apt update \
    && apt-get install -y git \
    && apt-get install -y libgl1-mesa-dev

RUN git clone --branch training https://github.com/mmvih/splinedist.git
RUN pip3 install cmake --no-deps
RUN pip3 install splinedist/ --no-deps

WORKDIR ${EXEC_DIR}
ENTRYPOINT ["python3", "main.py"]