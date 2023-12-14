FROM ubuntu:20.04 AS builder

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list &&\ 
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list &&\ 
    apt-get update && apt-get install -y python3-dev python3-pip &&\
    apt-get clean && rm -rf /tmp/* /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/*

# install Agilebot controller SDK
COPY ./Agilebot.IR.A-1.0-py3-none-any.whl .
RUN pip3 install --no-cache-dir Agilebot.IR.A-1.0-py3-none-any.whl



FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TIME_ZONE Asia/Shanghai
WORKDIR /home/adt

# update source and timezone
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list &&\
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list &&\ 
    apt-get update &&\
    apt-get install --no-install-recommends -y tzdata &&\
    ln -sf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && echo $TIME_ZONE > /etc/timezone &&\
    dpkg-reconfigure -f noninteractive tzdata &&\
    apt-get clean &&\
    rm -rf /tmp/* /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/*

# liboctomap-dev libfcl-dev &&\ 
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip gcc g++ &&\
    apt-get update && \
    apt-get install -y python3-dev &&\
    apt-get clean && \
    rm -rf /tmp/* /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/* &&\
    python3 -m pip install --no-cache-dir --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple

# copy Agilebot package
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages

RUN python3 -m pip install --no-cache-dir --upgrade \
    setuptools \
    APScheduler \
    pynng \
    cffi \
    numpy \
    tqdm \
    pandas \
    requests \
    google \
    protobuf \
    #python-fcl \
    toppra \
    protocol \
    grpcio \
    cython \
    aenum \
    -i https://pypi.mirrors.ustc.edu.cn/simple

