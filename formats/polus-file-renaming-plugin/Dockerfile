
FROM python:3.9-alpine

# Environment variables		
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"
# Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_LOG="INFO"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]
