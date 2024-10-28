import transformers
from functionary.prompt_template import PromptTemplate
from transformers import AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
import math
from torch.utils.data import DataLoader
import os
from typing import List

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


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
    """Preprocesses the logits during evaluation by computing the greedy token predictions for
    accuracy calculation and loss values for perplexity calculation. Both pred_ids and loss are
    of shape (batch_size x seq_len)"""

    correct_logits = logits
    if (
        type(logits) is tuple
    ):  # in mixtral logits is a tuple, correct logits is at the second index
        correct_logits = logits[1]

    pred_ids = torch.argmax(correct_logits, dim=-1)

    loss_fn = CrossEntropyLoss(reduction="none")
    shift_logits = correct_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, tokenizer_size)
    shift_labels = shift_labels.view(-1)
    loss = loss_fn(shift_logits, shift_labels)
    loss = torch.mean(loss.view(correct_logits.shape[0], -1), dim=-1)

    return pred_ids, loss


def compute_metrics(eval_preds, id2token, tokenizer):
    """Computes next-token accuracy and perplexity metrics for evaluation"""
    predictions = eval_preds.predictions[0][:, :-1]
    labels = eval_preds.label_ids[:, 1:]

    acc_count = 0
    total_num = 0
    dic = {token_id: {"acc": 0, "total": 0} for token_id in id2token}

    first_token_total_count, first_token_correct_count = 0, 0
    prediction_list, label_list = (
        predictions.flatten().tolist(),
        labels.flatten().tolist(),
    )
    first_token_label_dic = {}

    for i in range(len(prediction_list)):
        pred, label = prediction_list[i], label_list[i]
        if i > 0 and label_list[i - 1] == -100 and label != -100:  # first token
            first_token_total_count += 1
            if label not in first_token_label_dic:
                first_token_label_dic[label] = {"correct": 0, "total": 0}

            first_token_label_dic[label]["total"] += 1

            if label == pred:
                first_token_correct_count += 1
                first_token_label_dic[label]["correct"] += 1

        if label != -100:
            if label == pred:
                acc_count += 1
            total_num += 1
        if label in dic:
            dic[label]["total"] += 1
            if label == pred:
                dic[label]["acc"] += 1

    # Calculate perplexity
    loss = eval_preds.predictions[1].tolist()
    loss = sum(loss) / len(loss)
    perplexity = math.exp(loss)

    metrics = {
        "accuracy": acc_count / total_num,
        "perplexity": perplexity,
        "accuracy_first_token": first_token_correct_count / first_token_total_count,
        "total_number_first_token": first_token_total_count,
    }

    for token_id, stat in sorted(
        first_token_label_dic.items(), key=lambda x: -x[1]["total"]
    )[:5]:
        token = tokenizer.decode([token_id])
        metrics[f"accuracy_first_token_{token}"] = stat["correct"] / stat["total"]
        metrics[f"accuracy_first_token_{token}_total"] = stat["total"]

    for token_id in dic:
        token = id2token[token_id]
        total_num = dic[token_id]["total"]
        acc = -1
        if total_num > 0:
            acc = dic[token_id]["acc"] / total_num
        metrics[f"accuracy_{token}"] = acc
        metrics[f"accuracy_total_num_{token}"] = total_num

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
