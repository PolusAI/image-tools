FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".arrow"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
RUN pip3 install --index-url https://test.pypi.org/simple/ filepattern==2.2.7
COPY src ${EXEC_DIR}/src

RUN pip3 install ${EXEC_DIR} --no-cache-dir

ENTRYPOINT ["python3", "-m", "polus.images.formats.tabular_to_arrow"]
CMD ["--help"]
