import argparse
import logging

import sky

CLOUD_MAPPING = {
    "lambda": sky.Lambda(),
    "runpod": sky.RunPod(),
}


def get_cloud_provider(cloud_name: str) -> sky.clouds.Cloud:
    """
    Get the cloud provider object based on the given cloud name.

    Args:
        cloud_name (str): The name of the cloud provider.

    Returns:
        sky.clouds.Cloud: The corresponding cloud provider object.

    Raises:
        AssertionError: If an invalid cloud provider name is given.
    """
    assert cloud_name.lower() in CLOUD_MAPPING, f"Invalid cloud provider: {cloud_name}"
    return CLOUD_MAPPING[cloud_name.lower()]


def check_features(
    cloud: sky.clouds.Cloud, args: argparse.Namespace, logger: logging.Logger
):
    """
    Check if the cloud provider supports certain features and update arguments accordingly.

    This function checks if the given cloud provider supports stopping instances and opening ports.
    If these features are not supported, it updates the corresponding arguments and logs warnings.

    Args:
        cloud (sky.clouds.Cloud): The cloud provider object to check.

    Side effects:
        - May modify global 'args' object.
        - Logs warnings for unsupported features.
    """
    unsupported_features = cloud._unsupported_features_for_resources(None)

    if sky.clouds.CloudImplementationFeatures.STOP in unsupported_features:
        logger.warning(
            f"Stopping is not supported on {repr(cloud)}. Setting args.idle_timeout and args.down to None."
        )
        args.idle_timeout = None
        args.down = None
    if sky.clouds.CloudImplementationFeatures.OPEN_PORTS in unsupported_features:
        logger.warning(
            f"Opening port is not supported on {repr(cloud)}. Setting args.port_to_open to None. Please open port manually."
        )
        args.port_to_open = None


def form_setup(args: argparse.Namespace) -> str:
    """
    Form the setup command string for initializing the environment.

    This function constructs the setup command string that handles cloning the repository
    and checking out a specific commit if specified.

    Args:
        args (argparse.Namespace): The parsed command line arguments containing:
            - commit (str, optional): Git commit hash to checkout. If None, uses latest main branch.

    Returns:
        str: The formatted setup command string.
    """
    setup = "if [ ! -d 'functionary' ]; then git clone https://github.com/meetkai/functionary.git && cd functionary"
    if args.commit is not None:
        setup += f" && git checkout {args.commit}"
    setup += "; else cd functionary; fi && "

    return setup
