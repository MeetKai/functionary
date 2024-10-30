import argparse
import logging

import sky

from functionary.skypilot_utils import (
    CLOUD_MAPPING,
    check_features,
    form_setup,
    get_cloud_provider,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def form_command() -> str:
    """
    Form the command to run the vLLM server.

    This function constructs the command string to start the vLLM server
    based on the provided arguments. It includes the model, port, host,
    and optional parameters like max_model_len and tensor_parallel_size.

    Returns:
        str: The formatted command string to run the vLLM server.
    """
    command = "cd functionary && "
    if args.backend == "vllm":
        command += f"python server_vllm.py --model {args.model} --port {args.port} --host {args.host}"
        if args.max_model_len is not None:
            command += f" --max-model-len {args.max_model_len}"
    else:
        command += f"python server_sglang.py --model {args.model} --port {args.port} --host {args.host}"
        if args.max_model_len is not None:
            command += f" --context-length {args.max_model_len}"

    if args.tensor_parallel_size is not None:
        command += f" --tensor-parallel-size {args.tensor_parallel_size}"
    return command


def main():
    """
    Main function to deploy a Functionary model using Skypilot.

    This function performs the following steps:
    1. Retrieves the cloud provider based on the specified argument.
    2. Checks the features supported by the cloud provider.
    3. Creates a Skypilot Task with the necessary setup and run commands.
    4. Sets the resources for the task, including cloud, accelerators, ports, and disk size.
    5. Launches the task using Skypilot, with specified cluster name and optional timeout settings.

    Side effects:
        - Modifies global 'args' object based on cloud provider features.
        - Launches a Skypilot task, which may create or modify cloud resources.

    Raises:
        Any exceptions raised by Skypilot during task creation or launch.
    """
    cloud = get_cloud_provider(cloud_name=args.cloud)
    check_features(cloud=cloud, args=args, logger=logger)

    envs = {}

    setup = form_setup(args=args)
    if args.backend == "vllm":
        setup += "pip install -e .[vllm]"
    else:
        setup += "pip install -e .[sglang] --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/"

    # Authenticate HF if token is provided
    if args.hf_token:
        envs["HF_TOKEN"] = args.hf_token
        setup += f" && huggingface-cli login --token $HF_TOKEN"

    task = sky.Task(
        setup=setup,
        run=form_command(),
        envs=envs,
        workdir=None,
    )

    task.set_resources(
        sky.Resources(
            cloud=cloud,
            accelerators=f"{args.accelerators}:{args.num_accelerators}",
            ports=args.port_to_open,
            disk_size=args.disk_size,
            region=args.region,
        )
    )

    sky.launch(
        task,
        cluster_name=args.cluster_name,
        idle_minutes_to_autostop=args.idle_timeout,
        down=args.down,
        detach_run=args.detach_run,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy Skypilot")
    parser.add_argument(
        "--cluster-name", type=str, required=True, help="Name of the cluster"
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Provide a commit hash to deploy a specific version of Functionary. If None, the latest commit in the main branch will be deployed.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "sglang"],
        default="vllm",
        help="Backend inference framework to use. (Currently either `vllm` or `sglang`)",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default=None,
        help=f"Cloud provider (default: None). Currently only supports {list(CLOUD_MAPPING.keys())}",
    )
    parser.add_argument(
        "--accelerators",
        type=str,
        default="A100",
        help="Accelerator type. Check available types with `sky show-gpus --all`",
    )
    parser.add_argument(
        "--num-accelerators",
        type=int,
        default=1,
        help="Number of accelerators. Check available values with `sky show-gpus --all`",
    )
    parser.add_argument(
        "--disk-size",
        type=str,
        default=256,
        help="The size of the OS disk in GiB. If None, defaults to 256 GiB",
    )
    parser.add_argument(
        "--region", type=str, default=None, help="Region (default: None)"
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=-1,
        help="Idle timeout in minutes. `-1` means no timeout",
    )
    parser.add_argument(
        "--down",
        type=bool,
        default=False,
        help="Whether to tear down the cluster when timeout",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meetkai/functionary-small-v3.2",
        help="Model to use",
    )
    parser.add_argument("--max-model-len", type=int, default=None, help="Model to use")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to use")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host to use")
    parser.add_argument(
        "--detach-run",
        type=bool,
        default=True,
        help="Detach run upon job to run server is submitted.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for downloading models. Only use this is the model is gated or private.",
    )

    args = parser.parse_args()

    if args.disk_size is None:
        args.disk_size = 256
    args.disk_size = min(int(args.disk_size), 1024)  # Set max disk size to 1TB
    if args.idle_timeout == -1:
        args.idle_timeout = None
    args.port_to_open = args.port

    return args


if __name__ == "__main__":
    args = parse_args()
    main()
