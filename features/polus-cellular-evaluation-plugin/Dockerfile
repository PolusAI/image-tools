FROM labshare/polus-bfio-util:2.1.8

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN apt-get update
RUN apt-get install -y software-properties-common && \
    apt-get install -y libgl1-mesa-glx

RUN apt-get update && \
    pip3 install --default-timeout=100 --upgrade pip --no-cache-dir && \
    pip3 install --default-timeout=10000 --upgrade cython && pip3 install scipy

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir


ENTRYPOINT ["python3", "main.py"]