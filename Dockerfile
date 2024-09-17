FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04 

RUN apt-get update --yes --quiet
RUN apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y && \
     apt-get install -y python3.10 \
                        pip \
                        python3.10-distutils \
                        curl

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install -U pip

WORKDIR /app

COPY requirements.txt /app

RUN python3.10 -m pip install -r requirements.txt

COPY . /app

CMD python3.10 server_vllm.py --model "meetkai/functionary-small-v2.2" --host 0.0.0.0
