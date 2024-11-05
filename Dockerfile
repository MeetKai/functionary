# Use vLLM's vllm-openai server image as the base
FROM vllm/vllm-openai:v0.6.3.post1

# Define a build argument for the working directory, defaulting to /workspace
ARG WORKDIR_ARG=/workspace

# Set the working directory
WORKDIR ${WORKDIR_ARG}

# Install necessary build dependencies for sentencepiece
RUN apt-get update && apt-get install -y \
    pkg-config \
    cmake \
    build-essential

# Copy functionary code and requirements into workspace
COPY . .

# Install additional Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt

# Override the VLLM entrypoint with the functionary server
ENTRYPOINT ["python3", "server_vllm.py", "--model", "meetkai/functionary-small-v3.2", "--host", "0.0.0.0", "--max-model-len", "8192"]
CMD []
