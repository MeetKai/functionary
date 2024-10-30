import argparse
import logging
from typing import Any, Dict

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


def parse_train_command(train_command_file: str) -> dict:
    """
    Parse a training command file and extract arguments into a dictionary.

    This function reads a shell script containing training commands and parses the command line
    arguments into a dictionary format. It handles both flag arguments (--flag) and key-value
    arguments (--key value).

    Args:
        train_command_file (str): Path to the shell script containing the training command.

    Returns:
        dict: Dictionary mapping argument names (without -- prefix) to their values.
              For flag arguments, the value will be True.
              For key-value arguments, the value will be the string value provided.

    Example:
        For a command file containing:
        ```
        deepspeed train.py --model_name model1 --bf16 --epochs 3
        ```
        Returns:
        {
            'model_name': 'model1',
            'bf16': True,
            'epochs': '3'
        }
    """
    # Read train command file and parse arguments into dictionary
    train_args = {}
    with open(train_command_file, "r") as f:
        cmd = f.read()
        # Split on newlines and backslashes to get all parts
        parts = [p.strip() for p in cmd.replace("\\\n", " ").split()]
        for i, part in enumerate(parts):
            if part.startswith("--"):
                key = part[2:]  # Remove -- prefix
                # If this is the last argument or next is another flag, treat as boolean flag
                if i == len(parts) - 1 or parts[i + 1].startswith("--"):
                    train_args[key] = True
                else:
                    # Otherwise take the next part as the value
                    train_args[key] = parts[i + 1]
    return train_args


def form_command(train_file_args: Dict[str, Any]) -> str:
    """
    Form the command string for training and model uploading.

    This function constructs a multi-step command string that:
    1. Changes to the functionary directory
    2. Creates a training script with the provided arguments
    3. Runs the training script
    4. If using LoRA, merges the learned weights with the base model
    5. Creates a private repo on Hugging Face
    6. Uploads the trained model to Hugging Face

    Args:
        train_file_args (Dict[str, Any]): Dictionary of training arguments parsed from
            the training command file, containing keys like 'model_name_or_path',
            'output_dir', etc.

    Returns:
        str: The complete command string to execute all training and upload steps.
    """
    command = "cd functionary && "
    # Write the training command to a file to be run
    with open(args.train_command_file, "r") as f:
        train_command = f.read()
    command += "cat > train.sh << 'EOL'\n"
    command += train_command + " \\"
    command += f"\n    --train_data_path train.jsonl \\"
    command += f"\n    --eval_data_path val.jsonl"
    command += "\nEOL\n"
    command += "bash train.sh"
    # Merge the learned model with the base model
    if args.method == "lora":
        command += f" && python -m functionary.train.merge_lora_weight merged_model {train_file_args['model_name_or_path']} {train_file_args['output_dir']} {train_file_args['model_max_length']} {train_file_args['prompt_template_version']}"
        model_name = "merged_model"
    else:
        model_name = train_file_args["output_dir"]
    # Create new private repo on Hugging Face
    command += f' && python -c \'from huggingface_hub import create_repo; create_repo("{args.hf_organization}/{train_file_args["output_dir"]}", repo_type="model", private=True)\''
    # Push the model to the new repo
    command += f' && python -c \'from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(repo_id="{args.hf_organization}/{train_file_args["output_dir"]}", folder_path="{model_name}", repo_type="model")\''
    return command


def main():
    """
    Main function to train a Functionary model using Skypilot.

    This function performs the following steps:
    1. Gets the cloud provider based on the specified argument
    2. Checks the features supported by the cloud provider
    3. Parses the training command file to extract arguments
    4. Forms the setup command to initialize the environment with:
       - Moving data files into place
       - Installing PyTorch and other dependencies
       - Installing optional components (LoRA, Liger) if specified
       - Logging into Hugging Face and Weights & Biases if tokens provided
    5. Creates a Skypilot Task with the setup and training commands
    6. Sets the task resources including cloud, accelerators, and disk size
    7. Launches the task with specified cluster name and timeout settings

    The function handles both regular training and LoRA fine-tuning, with automatic
    model merging and uploading to Hugging Face Hub after training completes.

    Side effects:
        - Creates or modifies cloud resources via Skypilot
        - Uploads trained model to Hugging Face Hub
        - Logs training metrics to Weights & Biases if configured

    Raises:
        Any exceptions raised by Skypilot during task creation or launch.
    """
    cloud = get_cloud_provider(cloud_name=args.cloud)
    check_features(cloud=cloud, args=args, logger=logger)
    train_file_args = parse_train_command(args.train_command_file)

    envs = {}

    # Form setup command
    setup = form_setup(args=args)
    setup += (
        "mv ../train.jsonl ./train.jsonl && mv ../val.jsonl ./val.jsonl"
        " && cd functionary/train"
        " && pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121"
        " torchaudio==2.4.0+cu121"
        " --index-url https://download.pytorch.org/whl/cu121 && pip install -e ."
    )
    sections = []
    if args.method == "lora":
        sections.append("lora")
    if train_file_args.get("use_liger", False):
        sections.append("liger")
    if sections:
        setup += f"[{','.join(sections)}]"

    # Authenticate HF and WandB if tokens are provided
    if args.hf_token:
        envs["HF_TOKEN"] = args.hf_token
        setup += f" && huggingface-cli login --token $HF_TOKEN"
    if args.wandb_token:
        envs["WANDB_API_KEY"] = args.wandb_token
        setup += f" && wandb login $WANDB_API_KEY"

    # Define task with setup and run commands
    task = sky.Task(
        setup=setup,
        run=form_command(train_file_args=train_file_args),
        envs=envs,
        workdir=None,
    )

    # Set task resources and move data files into instance
    task.set_resources(
        sky.Resources(
            cloud=cloud,
            accelerators=f"{args.accelerators}:{args.num_accelerators}",
            disk_size=args.disk_size,
            image_id=args.runpod_image_id if isinstance(cloud, sky.RunPod) else None,
            region=args.region,
        )
    ).set_file_mounts(
        {
            "./train.jsonl": args.train_data_path,
            "./val.jsonl": args.eval_data_path,
        }
    )

    # Launch the task
    sky.launch(
        task,
        cluster_name=args.cluster_name,
        idle_minutes_to_autostop=args.idle_timeout,
        down=args.down,
        detach_run=args.detach_run,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy training via Skypilot")

    # Cluster arguments
    parser.add_argument(
        "--cluster-name", type=str, required=True, help="Name of the cluster"
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help=(
            "Provide a commit hash to deploy a specific version of Functionary. "
            "If None, the latest commit in the main branch will be deployed."
        ),
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
        "--runpod-image-id",
        type=str,
        default="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
        help="The image id to run the runpod instance on. (Only used if cloud is RunPod)",
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
        "--detach-run",
        type=bool,
        default=True,
        help="Detach run upon job to run server is submitted.",
    )

    # Training arguments
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "lora"],
        default="full",
        help="Training method to use. (Currently either `full` or `lora`)",
    )
    parser.add_argument(
        "--train-command-file",
        type=str,
        default="train.sh",
        help="Path to the command to train.",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to the local training data.",
    )
    parser.add_argument(
        "--eval-data-path",
        type=str,
        required=True,
        help="Path to the local validation data.",
    )

    # Uploading to Hugging Face and Weights & Biases
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for downloading models. Only use this is the model is gated.",
    )
    parser.add_argument(
        "--wandb-token",
        type=str,
        default=None,
        help="Wandb token for logging into wandb. Use this if you want to log into wandb.",
    )
    parser.add_argument(
        "--hf-organization",
        type=str,
        default="meetkai",
        help="Hugging Face organization to push the model to.",
    )

    args = parser.parse_args()

    if args.disk_size is None:
        args.disk_size = 256
    args.disk_size = min(int(args.disk_size), 1024)  # Set max disk size to 1TB
    if args.idle_timeout == -1:
        args.idle_timeout = None

    return args


if __name__ == "__main__":
    args = parse_args()
    main()
