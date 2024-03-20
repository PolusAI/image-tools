FROM polusai/bfio:2.3.6

# environment variables defined in polusai/bfio
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"
# Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_LOG="INFO"

ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV MIDRC_API_KEY="credentials.json"

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY CHANGELOG.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN pip3 install ${EXEC_DIR} --no-cache-dir

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "-m", "polus.images.utils.midrc_download"]
CMD ["--help"]
