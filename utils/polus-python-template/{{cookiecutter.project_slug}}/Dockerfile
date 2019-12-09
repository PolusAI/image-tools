{%- if cookiecutter.use_bfio %}
FROM labshare/polus-bfio-util:1.0.2
{%- else %}
FROM python:3.7-alpine
{%- endif %}

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

ENTRYPOINT ["python3", "main.py"]