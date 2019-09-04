FROM labshare/polyglot-notebook:0.2.0

ARG EXEC_DIR="/opt/executables"

#Copy executable
COPY src ${EXEC_DIR}/

# Install papermill
RUN pip install git+https://github.com/ktaletsk/papermill.git

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python", "execute_notebook.py"]