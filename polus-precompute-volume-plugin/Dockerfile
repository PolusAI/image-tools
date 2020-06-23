FROM labshare/polus-bfio-util:1.0.10-slim-buster

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

RUN python3 ${EXEC_DIR}/dl_fi.py

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
