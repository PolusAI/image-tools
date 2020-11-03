# Get Linux
FROM python:3.6.11-slim-buster

### Things that change rarely between containers

# Create temporary folder
ARG DATA_DIR="/data"
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}

# Copy the requirements file
COPY ./requirements-slim.txt ${EXEC_DIR}/

# Install bfio and requirements
RUN apt-get update && \
    pip3 install -r  ${EXEC_DIR}/requirements-slim.txt --no-cache-dir

### Things that change frequently between containers

# Copy bfio
COPY . ${EXEC_DIR}/

# Copy the version to the source directory
COPY VERSION ${EXEC_DIR}/src/

# Install bfio, remove the source directory
RUN pip3 install ${EXEC_DIR}/ --no-cache-dir && \
    rm -rf ${EXEC_DIR}
