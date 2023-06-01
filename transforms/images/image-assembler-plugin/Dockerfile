FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> de6ea1d (Update: update to new plugin standard:)

#Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_LOG="INFO" 

ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
<<<<<<< HEAD
=======
# ENV POLUS_LOG="INFO" # Change to WARNING for fewer logs, and DEBUG for debugging
>>>>>>> cbaf94c (update/image-assembler-plugin)
=======
>>>>>>> de6ea1d (Update: update to new plugin standard:)

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}

<<<<<<< HEAD
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
=======
COPY pyproject.toml ${EXEC_DIR}
>>>>>>> f5f7212 (fix: update to version 1.2.0-dev0)
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY CHANGELOG.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN pip3 install --index-url https://test.pypi.org/simple/ filepattern==2.2.7
RUN pip3 install ${EXEC_DIR} --no-cache-dir

# Default command. Additional arguments are provided through the command line
<<<<<<< HEAD
<<<<<<< HEAD
ENTRYPOINT ["python3", "/opt/executables/main.py"]
>>>>>>> cbaf94c (update/image-assembler-plugin)
=======
ENTRYPOINT ["python3", "-m", "polus.transforms.images.image_assembler"]
>>>>>>> f5f7212 (fix: update to version 1.2.0-dev0)
=======
ENTRYPOINT ["python3", "-m", "polus.plugins.transforms.images.image_assembler"]
CMD ["--help"]
>>>>>>> de6ea1d (Update: update to new plugin standard:)
