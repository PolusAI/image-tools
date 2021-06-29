FROM labshare/polus-bfio-util:1.0.7-slim-buster

COPY VERSION /

ARG EXEC_DIR="/opt/executables"

#Create folders
RUN mkdir -p ${EXEC_DIR}

#Copy executable
COPY src ${EXEC_DIR}/

RUN pip install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT [ "python3", "models.py" ]