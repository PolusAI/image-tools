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
COPY segmentation ${EXEC_DIR}/segmentation
RUN pip3 install ${EXEC_DIR}/segmentation/imagej-threshold-apply-tool --no-cache-dir

RUN mkdir ${EXEC_DIR}/transforms
RUN mkdir ${EXEC_DIR}/transforms/imagej-filter-frangivesselness-tool
WORKDIR ${EXEC_DIR}/transforms/imagej-filter-frangivesselness-tool

# Install the python package
COPY pyproject.toml .
COPY VERSION .
COPY README.md .
COPY src ./src
COPY tests ./tests

RUN pip3 install ${EXEC_DIR}/transforms/imagej-filter-frangivesselness-tool --no-cache-dir

RUN pip3 install scikit-image scipy pytest pytest-sugar pytest-json-report --no-cache-dir
RUN python3 -X faulthandler -m pytest --json-report --json-report-summary -v -p no:faulthandler
RUN cat .report.json

ENTRYPOINT ["python3", "-m", "polus.images.transforms.imagej_filter_frangivesselness"]
CMD ["--help"]
