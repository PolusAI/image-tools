
FROM tensorflow/tensorflow:2.4.1-gpu

# environment variables defined in labshare/polus-bfio-util
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"
# ENV POLUS_LOG="INFO" # Change to WARNING for fewer logs, and DEBUG for debugging

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}



COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/


RUN apt-get update && apt-get install --no-install-recommends -y g++ gcc build-essential && pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir


# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]
