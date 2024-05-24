import datetime
import math
import random
import sys
from typing import Any, Dict, List, Tuple

import monkey_patch_packing
import torch
import transformers
import typer
from datasets import load_dataset
from packed_dataset import PackedDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

random.seed(1)
torch.manual_seed(3)


def compute_loss_of_model(
    model: Any, ds: Dataset, tokenizer: Any, batch_size=8
) -> Tuple[float, int]:
    """Compute the avg loss per token given the model and dataset
    also return the number of tokens for computing loss

    Args:
        model (Any): model to compute the loss
        ds (Dataset): dataset to compute the loss
        tokenizer (Any): Tokenizer
        batch_size (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    total_loss = 0
    model.eval()

    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_num_loss_tokens = 0  # this is the total number of tokens for computing loss

    for index, batch in enumerate(data_loader):
        print(f"compute loss for batch: {index}")
        for key in batch:
            batch[key] = batch[key].to(model.device)

        if "labels" not in batch:
            labels = batch["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[labels == tokenizer.bos_token_id] = -100
            batch["labels"] = labels

        batch["return_dict"] = True

        with torch.no_grad():
            avg_loss = model.forward(**batch).loss.item()
            # compute number of tokens used for computing loss
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            ignore_count = (shift_labels == -100).sum()
            num_tokens = shift_labels.size(0) - ignore_count

            total_num_loss_tokens += num_tokens.item()
            total_loss += avg_loss * num_tokens.item()
    return total_loss / total_num_loss_tokens, total_num_loss_tokens


def compute_loss_for_model_class(
    pretrained_path: str, tokenizer: Any, ds: Any
) -> Tuple[float, int]:
    """Compute the loss with model initilized from model_class
        also return the number of tokens for computing the loss
    Args:
        pretrained_path (str): model_path
        model_class (Any): model_class to initialize model, can be monkey-patched class or original class
        tokenizer (Any): _description_
        ds (Any): _description_

    Returns:
        _type_: _description_
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # In the training, we set use_cache=False, use_cache=True only takes effect at inference
    model.config.use_cache = False

    model.config.output_router_logits = True  # force the model to compute loss

    if hasattr(model, "router_aux_loss_coef"):
        print("set au_coef=0")
        model.router_aux_loss_coef = (
            0  # excluding auxilary loss (in MoE model) in comparison
        )

    t1 = datetime.datetime.now()
    result = compute_loss_of_model(model, ds, tokenizer)
    t2 = datetime.datetime.now()
    print(f"time for computing the loss: {(t2 - t1).total_seconds()}")
    return result


def create_labels_from_input_ids(input_ids: List[int], tokenizer: Any) -> List[int]:
    """Mask all token_ids to -100 except token_ids of output

    Args:
        input_ids (List[int]): input_ids
        tokenizer (Any): tokenizer

    Returns:
        _type_: _description_
    """
    labels = list(input_ids)
    output_prefix = tokenizer.encode("\n### Response:", add_special_tokens=False)
    # Sometimes Llamatokenizer adds 29871 (in Llama2-model) and 28705 (in Mistal-model), we need to remove
    if output_prefix[0] in [28705, 29871]:
        output_prefix = output_prefix[1:]
    index = None  # find the index of output_prefix
    for i in range(len(input_ids)):
        if input_ids[i : i + len(output_prefix)] == output_prefix:
            index = i + len(output_prefix)
            break
    # Mask input_ids until token_ids of: "\n### Response:"
    for i in range(index):
        labels[i] = -100
    return labels


def main(
    pretrained_path: str,
    max_input_length: int = typer.Option(default=4096),
    pack_length: int = typer.Option(default=-1),
    masking_labels: bool = typer.Option(default=False),
    max_packed_size: int = typer.Option(default=-1),
):
    """This function is used to assert that the loss of monkey-patched models on packed datasets == loss of original models on original datasets
        We will use 50 random data points from dataset: tatsu-lab/alpaca on Huggingface Hub for computing the loss.
    Args:
        pretrained_path (str): model_path
        max_input_length (int, optional): max_length at tokenizing data. Defaults to 4096.
        pack_length (int, optional): The maximum length of packed data points, if = 1 --> value = max_input_length. Defaults to -1.
        masking_labels: whether we mask labels such that only Output tokens are used for computing loss:
            + masking_labels = True: masking prompt tokens as -100, and keep the output tokens --> only output tokens are used for computing loss
            + masking_labels = False: no masking, all tokens are used for computing loss
        max_packed_size (int, optional): Maximum number of data points that can be packed. If value = -1, there is no limit for this, as long as the length of packed data point < pack_length. Defaults to -1.
    Returns:
        _type_: _description_
    """
    if pack_length == -1:
        pack_length = max_input_length

    assert pack_length <= max_input_length
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = transformers.AutoConfig.from_pretrained(pretrained_path)
    config_type = type(model_config).__name__.lower()

    ds = load_dataset("tatsu-lab/alpaca")["train"]
    # extract 100 random data points from ds
    size = len(ds)
    indices = [i for i in range(size)]
    random.shuffle(indices)

    # We randomly select 50 data points for computing loss
    ex_ds = ds.select(indices[:50])
    print("number of data points: ", len(ex_ds))
    original_columns = ex_ds.column_names

    def process_data(examples):
        prompts = examples["text"]
        input_dic = tokenizer(
            prompts, max_length=max_input_length, padding="max_length", truncation=True
        )

        if masking_labels:
            # create labels by masking tokens from start to: "### Response:" as -100
            batch_input_ids = input_dic["input_ids"]
            batch_labels = []
            for input_ids in batch_input_ids:
                labels = create_labels_from_input_ids(input_ids, tokenizer)
                batch_labels.append(labels)
            input_dic["labels"] = batch_labels

        return input_dic

    ex_ds = ex_ds.map(process_data, batched=True, remove_columns=original_columns)
    ex_ds.set_format("torch")

    # convert ex_ds --> packed dataset
    packed_ds = PackedDataset(
        ex_ds, tokenizer, max_input_length, pack_length, max_packed_size
    )
    packed_ds.stat()

    # first compute the average loss of the original model on normal dataset (without packing)
    original_avg_loss, original_token_count = compute_loss_for_model_class(
        pretrained_path, tokenizer, ex_ds
    )
    print("original_loss: ", original_avg_loss)

    if "mistral" in config_type:
        print("model: Mistral ")
        monkey_patch_packing.monkey_patch_packing_mistral()
    elif "llama" in config_type:
        print("model: Llama ")
        monkey_patch_packing.monkey_patch_packing_llama()
    elif "mixtral" in config_type:
        print("model: Mixtral")
        monkey_patch_packing.monkey_patch_packing_mixtral()
    elif "phi3" in config_type:
        print("model: Phi3")
        monkey_patch_packing.monkey_patch_packing_phi3()
    else:
        print(
            f"{config_type} is not supported, currently we only support: Mistral, Mixtral, Llama"
        )
        sys.exit(1)

    # compute the loss on packed dataset with monkey-patched model
    mk_avg_loss, mk_token_count = compute_loss_for_model_class(
        pretrained_path, tokenizer, packed_ds
    )
    print("monkey-patched loss: ", mk_avg_loss)

    # Make sure that number of tokens used for computing loss are the same in original dataset and packed dataset
    assert (
        original_token_count == mk_token_count
    ), f"number of tokens for computing loss is different: original_token_count = {original_token_count}, mk_token_count={mk_token_count}"
    diff_loss = math.fabs(mk_avg_loss - original_avg_loss) / original_avg_loss
    print(
        f"original_loss: {original_avg_loss}, monkey-patched loss: {mk_avg_loss}, diff={diff_loss:2.4f}%"
    )


if __name__ == "__main__":
    typer.run(main)
