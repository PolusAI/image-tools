FROM labshare/polus-bfio-util:1.0.10-slim-buster

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/

RUN apt-get update && apt-get install -y libgl1-mesa-dev

RUN pip install cmake

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

RUN git clone --branch decomposition https://github.com/mmvih/neurogen.git --recurse-submodules

RUN pip install neurogen/

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
