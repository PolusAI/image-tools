
FROM labshare/polus-bfio-util:2.1.9
		
# environment variables defined in labshare/polus-bfio-util
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"

# RUN apt-get update && \
#     apt-get -y install libglib2.0-0

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]