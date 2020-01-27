# Get Windows
FROM openjdk:8-jdk

COPY VERSION /

		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

#Copy executable
COPY src ${EXEC_DIR}/

# Installs, clean, and update   
USER root

RUN apt-get update && apt-get install -y python3 \
    python3-pip
RUN apt-get install -y build-essential python-dev wget
RUN apt-get install -y python python-distribute python-pip
RUN pip3 install --upgrade pip --no-cache-dir
RUN pip3 install Cython==0.29.14
RUN pip3 install python_bioformats==1.5.2 javabridge==1.0.18 numpy==1.18.1 --no-cache-dir
RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]