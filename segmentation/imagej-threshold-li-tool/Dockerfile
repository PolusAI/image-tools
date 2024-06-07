FROM polusai/bfio:2.3.6-imagej

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# ImageJ setup
RUN python3 -c "import imagej; imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4')"

# Install the ImageJ Threshold Apply tool
COPY imagej-threshold-apply-tool ${EXEC_DIR}/imagej-threshold-apply-tool
RUN pip3 install ${EXEC_DIR}/imagej-threshold-apply-tool --no-cache-dir

RUN mkdir ${EXEC_DIR}/imagej-threshold-li-tool
WORKDIR ${EXEC_DIR}/imagej-threshold-li-tool

# Install the python package
COPY pyproject.toml .
COPY VERSION .
COPY README.md .
COPY src ./src
COPY tests ./tests

RUN pip3 install ${EXEC_DIR}/imagej-threshold-li-tool --no-cache-dir

RUN pip3 install scikit-image pytest pytest-sugar pytest-json-report --no-cache-dir
RUN python3 -X faulthandler -m pytest --json-report --json-report-summary -v -p no:faulthandler
RUN cat .report.json

ENTRYPOINT ["python3", "-m", "polus.images.segmentation.imagej_threshold_li"]
CMD ["--help"]
