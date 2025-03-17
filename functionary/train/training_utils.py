import os
from typing import List

import numpy as np
import transformers
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from functionary.prompt_template import PromptTemplate

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


# LEVENT OZBEK:
# Single GPU setup
# below is a parallelized dataloader with pinned memory for efficient data transfer between cpu and gpu
def create_data_loader(dataset, batch_size, num_workers=4):
    """dataLoader optimized for large datasets."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Parallel workers
        pin_memory=True,  # Efficient transfer to GPU
        shuffle=True,  # Shuffling to improve training generalization
    )


# LEVENT OZBEK:
# Multi GPU setup
# for large datasets
# using torch.distributed for syncing and managing processes
def create_distributed_data_loader(dataset, batch_size):
    """Create a distributed DataLoader."""
    sampler = DistributedSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )


# LEVENT OZBEK:
# for large data sets, it is important to cache tokenized sequences instead of re-tokenizing them repeatedly
def tokenize_and_cache(dataset, tokenizer, cache_dir):
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    return dataset.map(tokenize_function, batched=True, cache_file_name=cache_dir)


# LEVENT OZBEK:
# dynamically change the batch size according to the sequence count by setting a max token per batch threshold.
def dynamic_batch_size(dataset, max_tokens_per_batch, tokenizer):
    lengths = [len(tokenizer.encode(ex)) for ex in dataset]
    batch_sizes = [max_tokens_per_batch // length for length in lengths]
    return batch_sizes


def initialize_tokenizer(
    *,
    model: transformers.AutoModelForCausalLM,
    model_name_or_path: str,
    prompt_template: PromptTemplate,
    model_max_length: int,
    cache_dir: str,
):
    """Initialize tokenizer and add special tokens, resizing vocab and embedding"""
    # Mistral requires left padding due to the Sliding Window Attention mechanism
    if "mistral" in type(model).__name__.lower():
        print("model is mistral so padding_side=left")
        padding_side = "left"
    else:
        padding_side = "right"

    # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        legacy=True,
    )

    # Add special tokens
    tokenizer.pad_token = tokenizer.eos_token
    added_tokens = prompt_template.get_additional_tokens()
    special_tokens = {"additional_special_tokens": added_tokens}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    # add chat_template for tokenizer
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()
    print("tokenizer: ", tokenizer)

    # Resize embedding
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return tokenizer


def preprocess_logits_for_metrics(logits, labels, tokenizer_size):

    # Retains the same behavior for tuple logits (logits[1] if needed).
    if isinstance(logits, tuple):
        logits = logits[1]  # Handle tuple logits

    pred_ids = logits.argmax(dim=-1)
    loss_fn = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # avoiding unnecessary memory allocation by directly using CrossEntropyLoss over contiguous tensors without extensive reshaping.
    # keeping the original tensor shapes wherever possible to minimize memory footprint
    shift_logits = shift_logits.view(-1, tokenizer_size)
    shift_labels = shift_labels.view(-1)

    # Instead of computing loss on a flattened vector and reshaping it back, just compute the loss batch-wise directly:
    loss = loss_fn(shift_logits, shift_labels).view(logits.size(0), -1).mean(dim=1)

    return pred_ids, loss


# LEVENT OZBEK:
# Optimized compute metrics for large data sets!
"""
- Replaced loops over sequences with matrix operations.
- NumPy for efficient batch-level calculations.
- Moved evaluation computations to the GPU when possible.
"""


def compute_metrics(eval_preds, id2token, tokenizer):
    # Predictions and labels
    predictions = eval_preds.predictions[0][:, :-1]
    labels = eval_preds.label_ids[:, 1:]

    # Mask to ignore padding tokens (-100 in labels)
    valid_mask = labels != -100

    # Computing accuracy
    correct_preds = (predictions == labels) & valid_mask
    acc_count = correct_preds.sum()
    total_num = valid_mask.sum()

    # Token-wise accuracy/batch processing
    token_stats = {
        token_id: {
            "acc": correct_preds[labels == token_id].sum().item(),
            "total": (labels == token_id).sum().item(),
        }
        for token_id in id2token.keys()
    }

    # Calculating perplexity
    losses = eval_preds.predictions[1]
    perplexity = np.exp(np.mean(losses))

    # First token-specific accuracy
    first_token_mask = (labels[:, :-1] == -100) & valid_mask[:, 1:]
    first_token_correct = (correct_preds[:, 1:] & first_token_mask).sum()
    first_token_total = first_token_mask.sum()

    metrics = {
        "accuracy": acc_count / total_num,
        "perplexity": perplexity,
        "accuracy_first_token": first_token_correct / max(first_token_total, 1),
    }

    # Token-specific accuracies
    for token_id, stat in token_stats.items():
        token = id2token[token_id]
        metrics[f"accuracy_{token}"] = stat["acc"] / max(stat["total"], 1)
        metrics[f"accuracy_total_num_{token}"] = stat["total"]

    return metrics


def extract_unmasked_chunks(labels: List[int], masked_value) -> List[List[int]]:
    """This function is used to extract unmasked chunks of integer
    For example, labels = [-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
    Args:
        labels (List[int]): list of integer containing token_id and -100

    Returns:
        List[List[int]]: list of chunk, for example: [[1,2,3], [4,5]]
    """
    chunks = []
    chunk = []
    for token_id in labels:
        if token_id != masked_value:
            chunk.append(token_id)
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
                chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    return chunks


def print_some_examples(ds, tokenizer):
    data_loader = DataLoader(ds, batch_size=3)
    count = 0
    for batch in data_loader:
        if count == 0:
            print_rank0("keys in batch: ", batch.keys())
        print_rank0("--------------****Example data point****---------------")
        print_rank0("device: ", batch["input_ids"].device)
        print_rank0("shape of input_ids: ", batch["input_ids"].shape)  # B x L
        print_rank0("shape of labels: ", batch["labels"].shape)
        print_rank0("shape of attention_mask: ", batch["attention_mask"].shape)
        # print_rank0('input_ids: ', batch["input_ids"].tolist())
        # print_rank0('labels: ', batch["labels"].tolist())
        print_rank0("attention mask: ", batch["attention_mask"])
        input_ids = batch["input_ids"][0].tolist()
        input_chunk = extract_unmasked_chunks(input_ids, tokenizer.pad_token_id)
        # assert len(input_chunk) == 1  # padding at left or right only --> pad_token_id = eos_token_id --> wrong
        print_rank0("+ inputs: ")
        print_rank0(tokenizer.decode(input_chunk[0]))
        labels = batch["labels"][0].tolist()
        label_chunks = extract_unmasked_chunks(labels, -100)
        print_rank0("----------")
        for chunk in label_chunks:
            print_rank0("+ chunk: ")
            print_rank0(tokenizer.decode(chunk))
        count += 1
        if count == 5:
            break
