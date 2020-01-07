FROM labshare/polus-bfio-util:1.0.4
LABEL maintainer="Axle Informatics"

COPY VERSION /

ENV DEBIAN_FRONTEND noninteractive
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

ENV OPENCV_VERSION=4.1.2

ENV ENABLE_HEADLESS=1

RUN mkdir /opt/cv2 && cd /opt/cv2 && \
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm -rf ${OPENCV_VERSION}.zip

RUN apk --update add --virtual build-dependencies git linux-headers python3-dev build-base wget cmake && \
    pip install git+https://github.com/skvark/opencv-python.git@fd9fb99f2bdbe6e16c5bf1832bb8db5d0af65249 && \
    apk del build-dependencies

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]