FROM python:3.6-alpine

COPY VERSION /

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir -p ${DATA_DIR}/outputs

#Copy executable
COPY src ${EXEC_DIR}/

# Install requirements
RUN apk --update add --virtual build-dependencies gcc gfortran python3-dev build-base wget freetype-dev libpng-dev openblas-dev \
    && pip3 install --upgrade cython \
    && pip3 install -r  ${EXEC_DIR}/requirements.txt --no-cache-dir \
    #&& pip3 install ${EXEC_DIR}/ --no-cache-dir \
    && rm -rf /root/.cache \
    && apk del build-dependencies

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]