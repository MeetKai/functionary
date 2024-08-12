# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.4.1
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS base

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && if [ "${PYTHON_VERSION}" != "3" ]; then update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; fi \
    && python3 --version

RUN apt-get update -y \
    && apt-get install -y git curl sudo

# Install pip s.t. it will be compatible with our PYTHON_VERSION
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
RUN python3 -m pip --version

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /app

COPY requirements.txt /app

RUN python3.10 -m pip install -r requirements.txt
# RUN pip show torch && sleep 60
COPY . /app

# CMD python3.10 server_vllm.py --model "hivata/functionary-small-v2.1-AWQ" --host 0.0.0.0 --trust-remote-code
ENTRYPOINT [ "python3.10", "server_vllm.py" ]
# CMD [ "pip", "show", "torch", "transformers", "vllm" ]
