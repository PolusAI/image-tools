FROM labshare/polus-bfio-util:2.1.9-tensorflow


COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/

RUN apt-get update

RUN apt install -y libgl1-mesa-glx

RUN pip3 install opencv-python-headless==4.5.3.56

RUN python3 -m pip install -r ${EXEC_DIR}/requirements.txt
                                                                                                                                                                                            

ENTRYPOINT ["python3", "main.py"]