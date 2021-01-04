{%- if cookiecutter.bfio_version.lower()=="none" %}
FROM labshare/polus-bfio-util:{{ cookiecutter.bfio_version }}{% if cookiecutter.bfio_container != '' %}-{{ cookiecutter.bfio_container }}{% endif %}
{%- else %}
FROM python:3.8-alpine
{%- endif %}

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]