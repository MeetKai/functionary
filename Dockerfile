# Use vLLM's vllm-openai server image as the base
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /workspace

# Install necessary build dependencies for sentencepiece
RUN apt-get update && apt-get install -y \
    pkg-config \
    cmake \
    build-essential

# Copy functionary code and requirements into workspace
# Clone functionary repository
RUN git clone https://github.com/MeetKai/functionary.git . && \
    cd functionary



# Install additional Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

# Authenticate HF
COPY huggingface_token.txt /tmp/huggingface_token.txt
RUN huggingface-cli login --token $(cat /tmp/huggingface_token.txt)
RUN rm /tmp/huggingface_token.txt

# Override the VLLM entrypoint with the functionary server
ENTRYPOINT ["python3", "server_vllm.py"]

CMD []