FROM labshare/polus-bfio-util:1.0.4-slim-buster

COPY VERSION /

RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.1.0-cp37-cp37m-manylinux2010_x86_64.whl keras opencv-python-headless matplotlib

ARG EXEC_DIR="/opt/executables"

#Create folders
RUN mkdir -p ${EXEC_DIR}

#Copy executable
COPY src ${EXEC_DIR}/

ENTRYPOINT [ "python3", "models.py" ]