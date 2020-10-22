FROM python:3.8-alpine

COPY VERSION /

# Create temporary folder
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir -p ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/

# Install requirements
RUN apk --update add --virtual build-dependencies g++ gcc gfortran python3-dev build-base wget freetype-dev libpng-dev openblas-dev \
    && pip3 install --upgrade cython \
    && pip3 install -r  ${EXEC_DIR}/requirements.txt --no-cache-dir \
    && rm -rf ${EXEC_DIR} \
    && apk del build-dependencies

WORKDIR ${EXEC_DIR}


ENTRYPOINT ["python3", "src/main.py"]
