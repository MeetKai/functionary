import os

import typer
import wandb


def main(entity: str, project: str, run_id: str, model_dir: str):
    """Uploads the final model as an artifact to the specific training run"""
    # Get the wandb run
    # run = wandb_api.run(f"{entity}/{project}/{run_id}")
    run = wandb.init(id=run_id, project=project, entity=entity, resume="allow")

    # Create a new artifact object to upload the model to
    artifact = wandb.Artifact(name="final_model", type="model")

    # Add the model to the artifact's contents
    for filename in os.listdir(model_dir):
        file_path = os.path.join(model_dir, filename)

        if not os.path.isdir(file_path):
            artifact.add_file(local_path=file_path)

    # Save the artifact version to W&B and mark it
    # as the output of run_id
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    typer.run(main)
