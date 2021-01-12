# Compile ftl.pyx
FROM python:3.6-alpine
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY src ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}
RUN apk --update add --virtual build-dependencies python3-dev build-base wget \
    && pip3 install --upgrade cython==3.0a1 \
    && pip3 install numpy==1.18.1 --no-cache-dir \
    && python3 setup.py build_ext --inplace \
    && apk del build-dependencies

# Build the plugin container
FROM labshare/polus-bfio-util:1.0.10

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