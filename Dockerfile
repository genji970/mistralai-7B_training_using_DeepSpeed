FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && \
    apt install -y software-properties-common git curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1

RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install deepspeed transformers datasets accelerate peft bitsandbytes

WORKDIR /workspace
