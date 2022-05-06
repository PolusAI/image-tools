FROM labshare/polus-bfio-util:2.1.9-tensorflow
ENV EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/
RUN apt-get update && apt-get install --no-install-recommends -y g++ gcc build-essential && pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir
ENTRYPOINT ["python3", "main.py"]
