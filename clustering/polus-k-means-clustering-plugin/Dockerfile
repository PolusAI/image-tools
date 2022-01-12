FROM labshare/polus-bfio-util:2.1.9

# Change to .feather to save output images as feather files.
ENV POLUS_EXT=".feather"

# Change to WARNING for fewer logs, or DEBUG for debugging.
ENV POLUS_LOG="INFO"

COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]