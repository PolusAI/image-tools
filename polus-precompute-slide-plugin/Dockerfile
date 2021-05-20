# Get image containing bfio
FROM labshare/polus-bfio-util:2.0.6-slim-buster

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

#Copy executable
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

RUN python3 ${EXEC_DIR}/dl_fi.py

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]
