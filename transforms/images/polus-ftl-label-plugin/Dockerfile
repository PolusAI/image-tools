# Compile ftl.pyx
FROM python:3.9
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY src ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}
RUN pip3 install --upgrade cython==3.0a6 \
    && pip3 install numpy==1.19.1 --no-cache-dir \
    && python3 setup.py build_ext --inplace

# Build the plugin container
FROM python:3.9-slim

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY --from=0 /opt/executables ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]