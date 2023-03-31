FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"
<<<<<<< HEAD

#Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_LOG="INFO" 

ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
=======
# ENV POLUS_LOG="INFO" # Change to WARNING for fewer logs, and DEBUG for debugging
>>>>>>> cbaf94c (update/image-assembler-plugin)

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}

<<<<<<< HEAD
COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY CHANGELOG.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN pip3 install ${EXEC_DIR} --no-cache-dir

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "-m", "polus.plugins.transforms.images.image_assembler"]
CMD ["--help"]
=======
COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "/opt/executables/main.py"]
>>>>>>> cbaf94c (update/image-assembler-plugin)
