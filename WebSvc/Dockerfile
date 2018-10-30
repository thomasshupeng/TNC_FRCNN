FROM ubuntu:16.04
MAINTAINER Shu Peng <thomasshupeng@hotmail.com>

WORKDIR /code
COPY ./requirements.txt /code
COPY ./run_waitress_server.py /code
COPY ./tncApp.py /code
COPY ./hello.py /code
COPY ./BU_ModelLoader.py /code
COPY ./FasterRCNN/. /code/FasterRCNN/
COPY ./models/. /code/models/
COPY ./utils/annotations/. /code/utils/annotations
COPY ./utils/configs/. /code/utils/configs/
COPY ./utils/rpn/. /code/utils/rpn/
COPY ./utils/cython_modules/cpu_nms.cpython-35m-x86_64-linux-gnu.so /code/utils/cython_modules/cpu_nms.so
COPY ./utils/cython_modules/cython_bbox.cpython-35m-x86_64-linux-gnu.so /code/utils/cython_modules/cython_bbox.so
COPY ./utils/*.py /code/utils/

RUN apt-get update && apt-get install -y gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        openmpi-bin \
        libopenmpi-dev \
        cmake \
        libgtk2.0-dev \
        pkg-config \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        libpython3.5-dev \
        python3 \
        python3-dev \
        python3-setuptools \
        python3-pip \
        wget

RUN mkdir /usr/local/mklml
RUN wget https://github.com/01org/mkl-dnn/releases/download/v0.14/mklml_lnx_2018.0.3.20180406.tgz
RUN tar -xzf mklml_lnx_2018.0.3.20180406.tgz -C /usr/local/mklml
RUN wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.14.tar.gz | tar -xzf - && \
    cd mkl-dnn-0.14 && \
    ln -s /usr/local external && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf mkl-dnn-0.14

RUN pip3 install --upgrade pip
RUN pip3 install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.6-cp35-cp35m-linux_x86_64.whl
RUN pip3 install -r ./requirements.txt

RUN export LD_LIBRARY_PATH=/usr/local/opencv-3.1.0/lib:$LD_LIBRARY_PATH

EXPOSE 8080
ENV SERVER_PORT 8080

CMD ["python3", "run_waitress_server.py"]
