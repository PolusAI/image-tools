FROM polusai/bfio:2.1.9

COPY VERSION /
COPY src ${EXEC_DIR}/.

RUN apt --no-install-recommends -y autoremove --purge python3.9-minimal python3.9\
    && apt-get update && apt-get install --no-install-recommends -y build-essential python3.9-dev\
    && pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "/opt/executables/main.py"]