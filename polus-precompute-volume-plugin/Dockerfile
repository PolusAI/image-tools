FROM python:3.7
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}

COPY src ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}

RUN git clone --branch cache https://github.com/mmvih/neurogen.git --recurse-submodules
RUN pip3 install cmake
RUN pip3 install neurogen/


FROM python:3.7-slim

# Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_LOG="DEBUG" 

COPY VERSION /
ARG EXEC_DIR="/opt/executables"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

RUN apt-get update \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get install -y libgomp1

COPY --from=0 /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY --from=0 /opt/executables ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}
RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]