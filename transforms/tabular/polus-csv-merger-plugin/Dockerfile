FROM labshare/polus-bfio-util:2.1.9

# environment variables defined in labshare/polus-bfio-util
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"

# Change to .feather to save output file as feather file format.
# ENV POLUS_EXT=".feather"

# Change to WARNING for fewer logs, or DEBUG for debugging.
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "/opt/executables/main.py"]