FROM alpine

COPY VERSION /

# Install python
RUN apk --update --no-cache add python3 py3-pip curl ca-certificates && \
    pip3 install --upgrade pip --no-cache-dir

# Create temporary folder
ARG EXEC_DIR="/opt/executables"

RUN mkdir -p ${EXEC_DIR}

COPY src ${EXEC_DIR}/

# Install requirements
RUN apk --update add --virtual build-dependencies g++ gcc gfortran python3-dev build-base wget freetype-dev libpng-dev openblas-dev \
    && pip3 install --upgrade cython \
    && pip3 install -r  ${EXEC_DIR}/requirements.txt --no-cache-dir \
    && rm -rf ${EXEC_DIR} \
    && apk del build-dependencies
		
ARG DATA_DIR="/data"

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]

