FROM labshare/polus-bfio-util:2.1.9

COPY requirements.txt .

RUN pip3 install -r requirements.txt --no-cache-dir

COPY VERSION /
COPY src ${EXEC_DIR}/

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
