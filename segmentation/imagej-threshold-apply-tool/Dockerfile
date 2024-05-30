FROM polusai/bfio:2.3.6-imagej

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src
COPY tests ${EXEC_DIR}/tests

RUN pip3 install ${EXEC_DIR} --no-cache-dir
RUN python3 -c "import imagej; imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4')"

RUN pip3 install pytest pytest-sugar pytest-json-report --no-cache-dir
RUN python3 -X faulthandler -m pytest --json-report --json-report-summary -v -p no:faulthandler
RUN cat .report.json

ENTRYPOINT ["python3", "-m", "polus.images.segmentation.imagej_threshold_apply"]
CMD ["--help"]
