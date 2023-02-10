FROM polusai/bfio:2.1.9-tensorflow

COPY VERSION /

ARG EXEC_DIR="/opt/executables"

#Create folders
RUN mkdir -p ${EXEC_DIR}

#Copy executable
COPY src ${EXEC_DIR}/

RUN pip install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT [ "python3", "/opt/executables/models.py" ]