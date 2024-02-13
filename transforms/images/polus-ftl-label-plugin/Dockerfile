# Compile ftl.pyx
FROM python:3.9
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}/executables
COPY src ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}
RUN pip3 install --upgrade cython==3.0.0a9 \
    && pip3 install numpy==1.21.4 --no-cache-dir \
    && python3 setup.py build_ext --inplace
RUN cp executables/* . && rm -rf executables

# Build the plugin container
FROM python:3.9-slim

# Get essential packages
RUN apt-get update && \
    apt-get install -y build-essential curl && \
    apt-get update

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

ARG EXEC_DIR="/ftl-rust"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR}/{src, ftl_rust} && mkdir -p ${DATA_DIR}/{inputs, outputs}

COPY . ${EXEC_DIR}/

COPY --from=0 /opt/executables ${EXEC_DIR}/src

WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/src/requirements.txt --no-cache-dir \
    && pip3 install -r rust_requirements.txt --no-cache-dir \
    && python3 rust_setup.py install

WORKDIR ${EXEC_DIR}/src

ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

RUN ls -la ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]
