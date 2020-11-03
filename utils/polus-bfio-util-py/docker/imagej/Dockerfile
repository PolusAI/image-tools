# Get Linux
FROM maven:3-jdk-8-slim

### Things that change rarely between containers

# Copy Python 3.6
COPY --from=python:3.6 / /

# Create temporary folder
ARG DATA_DIR="/data"
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}

# Copy the requirements file
COPY ./requirements-slim.txt ${EXEC_DIR}/

# Install requirements
RUN apt-get update && \
    pip3 install -r  ${EXEC_DIR}/requirements-slim.txt --no-cache-dir

# Install packages
ENV LIBRARY_PATH=/lib:/usr/lib
RUN pip3 install cython --no-cache-dir \
    && python3 -m pip install git+https://github.com/imagej/pyimagej.git@6b6b4a8cf57824a4e42fcac6348b06db843c4522#egg=imagej --no-cache-dir

# Get PyJnius jar
RUN curl -L https://anaconda.org/conda-forge/pyjnius/1.2.0/download/linux-64/pyjnius-1.2.0-py37h93af967_0.tar.bz2 | tar xjf - share/pyjnius/pyjnius.jar
ENV PYJNIUS_JAR="/share/pyjnius/pyjnius.jar"

### Things that change frequently between containers

# Copy bfio
COPY . ${EXEC_DIR}/

# Copy the version to the source directory
COPY VERSION ${EXEC_DIR}/src/

# Install bfio, remove the source directory
RUN pip3 install ${EXEC_DIR}/ --no-cache-dir \
    && rm -rf ${EXEC_DIR}

RUN python3 -c 'import imagej; ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4"); print(ij.getVersion())'