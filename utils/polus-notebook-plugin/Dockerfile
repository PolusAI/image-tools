FROM labshare/polyglot-notebook:0.3.2

COPY VERSION /

ARG EXEC_DIR="/opt/executables"

#Copy executable
COPY src ${EXEC_DIR}/

# Install papermill
RUN pip install jupyter-core==4.6.3 \
    && pip install jupyter-client==6.1.6 \
    && pip install nbformat==5.0.7 \
    && pip install papermill==2.1.2 \
    && pip install git+https://github.com/vatlab/sos-papermill.git@dd2046e34c854fcaa90bba4bd7c604c7e9a6ad4f

WORKDIR ${EXEC_DIR}

USER root

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python", "execute_notebook.py"]