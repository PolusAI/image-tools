FROM labshare/polus-bfio-util:2.1.9
COPY VERSION /
COPY src ${EXEC_DIR}/.
RUN apt --no-install-recommends -y autoremove --purge python3.9-minimal python3.9
RUN apt-get update && apt-get install --no-install-recommends -y g++ gcc gfortran build-essential libpython3.9-stdlib libpython3-dev python3 python3.9-dev python3-pip
RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir
RUN rm -rf /root/.cache
ENTRYPOINT ["python3", "main.py"]


