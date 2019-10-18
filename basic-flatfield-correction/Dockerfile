FROM labshare/polus-bfio-util:1.0.1
LABEL maintainer="Axle Informatics"

COPY VERSION /

ENV DEBIAN_FRONTEND noninteractive
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

# Install opencv from source

# ENV CC /usr/bin/clang
# ENV CXX /usr/bin/clang++

ENV OPENCV_VERSION=4.1.2

ENV ENABLE_HEADLESS=1

RUN mkdir /opt/cv2 && cd /opt/cv2 && \
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm -rf ${OPENCV_VERSION}.zip

RUN apk --update add --virtual build-dependencies git linux-headers python3-dev build-base wget cmake && \
    pip install git+https://github.com/skvark/opencv-python.git@fd9fb99f2bdbe6e16c5bf1832bb8db5d0af65249 && \
    apk del build-dependencies

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

#Copy executable and version
COPY src ${EXEC_DIR}/

# RUN echo 'manylinux1_compatible = True' > /usr/lib/python3.6/site-packages/_manylinux.py \
#     && python3 -c 'import sys; sys.path.append(r"/_manylinux.py")' \
#     && pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "main.py"]