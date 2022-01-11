# Get image containing bfio
FROM labshare/polus-bfio-util:2.1.9

# set environment variables
ENV POLUS_LOG="INFO"
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"
# uses java backend
# do not specify polus_ext because it can handle .tif images too, not just .ome.tif

COPY VERSION /
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]