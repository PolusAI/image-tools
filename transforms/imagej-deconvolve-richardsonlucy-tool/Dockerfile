FROM polusai/bfio:2.3.6-imagej

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# Initialize ImageJ
RUN python3 -c "import imagej; imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4')"

# TODO: Change the tool_dir to the tool directory
ENV TOOL_DIR="transforms/imagej-deconvolve-richardsonlucy-tool"

# Copy the repository into the container
RUN mkdir image-tools
COPY . ${EXEC_DIR}/image-tools

# Install the tool
RUN pip3 install "${EXEC_DIR}/image-tools/${TOOL_DIR}" --no-cache-dir

# Set the entrypoint
# TODO: Change the entrypoint to the tool entrypoint
ENTRYPOINT ["python3", "-m", "polus.images.transforms.imagej_deconvolve_richardsonlucy"]
CMD ["--help"]
