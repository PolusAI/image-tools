### 1. Get Linux
FROM openjdk:8-alpine

COPY VERSION /

# Install python
RUN apk --update --no-cache add python3 ca-certificates && \
    pip3 install --upgrade pip --no-cache-dir

# Create temporary folder
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}

# Copy bfio
COPY . ${EXEC_DIR}/

# Install bfio and requirements
RUN apk --update add --virtual build-dependencies python3-dev build-base wget \
    && pip3 install python_bioformats==1.5.2 numpy==1.16.2 javabridge==1.0.18 --no-cache-dir \
    && pip3 install ${EXEC_DIR}/ --no-cache-dir \
    && rm -rf ${EXEC_DIR} \
    && apk del build-dependencies
		
ARG DATA_DIR="/data"

WORKDIR ${EXEC_DIR}