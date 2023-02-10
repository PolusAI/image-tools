FROM polusai/bfio:2.1.9

ENV EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/
RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "/opt/executables/main.py"]
