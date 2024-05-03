from typing import Any

import modal
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

gpu_config_mapping = {
    "meetkai/functionary-small-v2.4": modal.gpu.L4(count=1),
    "meetkai/functionary-medium-v2.4": modal.gpu.A100(count=2, size="80GB"),
    "meetkai/functionary-small-v2.2": modal.gpu.L4(count=1),
    "meetkai/functionary-medium-v2.2": modal.gpu.A100(count=2, size="80GB"),
    "meetkai/functionary-7b-v2.1": modal.gpu.L4(count=1),
    "meetkai/functionary-7b-v2": modal.gpu.L4(count=1),
    "meetkai/functionary-7b-v1.4": modal.gpu.L4(count=1),
}


class Settings(BaseSettings):
    model: str = "meetkai/functionary-small-v2.4"
    max_model_length: int = 8192
    gpu_config: Any = None  # Define the type according to what modal.gpu.* returns
    gpu_memory_utilization: float = 0.9
    storage_model_dir: str = "model"
    execution_timeout: int = 1200  # Generation timeout in seconds
    container_idle_timeout: int = 1200  # In seconds
    batch_size_per_container: int = 10
    enable_grammar_sampling: bool = False

    model_config = SettingsConfigDict(env_prefix="modal_")

    @field_validator("gpu_config")
    def set_gpu_config(cls, v, values):
        model = values.data.get("model")
        if model not in gpu_config_mapping:
            raise ValueError(f"GPU configuration for model {model} not found")
        return gpu_config_mapping[model]
