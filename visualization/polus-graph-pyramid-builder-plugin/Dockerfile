# Get image containing bfio
FROM python:3.9-slim

COPY VERSION /
		
ENV POLUS_LOG="INFO"
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

#Copy executable
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]
