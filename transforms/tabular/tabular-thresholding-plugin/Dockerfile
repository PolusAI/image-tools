FROM labshare/polus-bfio-util:2.1.9

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

ENV FILE_EXT="csv"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/{inputs, outputs}

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN apt-get -y update && \
    apt-get -y install git && \
    pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir --force

ENTRYPOINT ["python3", "main.py"]