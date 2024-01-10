FROM labshare/polus-bfio-util:2.1.9-tensorflow
ENV EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/
RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir && \
    pip3 install "bfio[all]"
ENTRYPOINT ["python3", "main.py"]