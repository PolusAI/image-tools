FROM polusai/bfio:2.1.9

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} && mkdir -p ${DATA_DIR}/{input, output}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY CHANGELOG.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN pip3 install ${EXEC_DIR} --no-cache-dir

WORKDIR ${EXEC_DIR}
ENTRYPOINT ["python3", "-m", "polus.plugins.transforms.images.rolling_ball"]
CMD ["--help"]
