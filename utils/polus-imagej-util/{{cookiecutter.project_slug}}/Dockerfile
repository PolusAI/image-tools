FROM labshare/polus-bfio-util:2.1.9-imagej

# environment variables defined in labshare/polus-bfio-util
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"
# ENV POLUS_LOG="INFO" # Change to WARNING for fewer logs, and DEBUG for debugging

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir
RUN mkdir /.jgo && chmod 777 /.jgo

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]