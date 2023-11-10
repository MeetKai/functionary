import transformers
from transformers import LlamaTokenizerFast
from functionary.train import custom_datasets
from functionary.prompt import get_additional_tokens
from functionary.train.monkey_patch.mistral_monkey_patched import MistralForCausalLM as MonkeyPatchedMistral
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import copy
import typer
import math
from typing import Optional, List, Any
from torch.utils.data import Dataset, DataLoader
import json 
from torch.nn import CrossEntropyLoss


class OriginalMistral(transformers.MistralForCausalLM):
    """Monkey-patch to add loss_reduction to def forward, this is to compute sum of the loss in tead of mean of the loss
    This is used for comparing the total loss between Normal Dataset and Packed Dataset

    Args:
        transformers (_type_): _description_
    """
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_reduction: str = "mean",
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction=loss_reduction)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            # print out tokens for computing loss:
            # loss_labels = []
            # for label in shift_labels.tolist():
            #     if label != -100:
            #         loss_labels.append(label)
            # print("loss_labels: ", loss_labels)
            # print("inside number of labels: ", len(loss_labels))
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_data_points():
    return [
        {
            "messages": [
                {"role": "user", "content": "hi, how are you"},
                {"role": "assistant", "content": "I am fine, thank you and you?"},
                {"role": "user", "content": "Oh I am good, where do you live now"},
                {"role": "assistant", "content": "I live in Hanoi"},
            ],
            "functions": [],
        },
        {
            "messages": [
                {"role": "user", "content": "this is a test"},
                {"role": "assistant", "content": "Oh I know that"},
                {"role": "user", "content": "you are smart"},
                {"role": "assistant", "content": "yes I am "},
            ],
            "functions": [],
        },
    ]


def compute_loss_from_ds_by_model(ds: Dataset, model, batch_size: int):
    total_loss = 0
    model.eval()
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for index, batch in enumerate(data_loader):
        print(f"compute loss for batch: {index}")
        for key in batch:
            batch[key] = batch[key].to(model.device)
            
        batch["return_dict"] = True
        batch["loss_reduction"] = "sum"
        
        with torch.no_grad():
            loss = model.forward(**batch).loss.item()
            total_loss += loss
    return total_loss 


def compute_loss_from_ds(ds: Dataset, pretrained_path, model_class: Any, num_new_tokens: int, tokenizer_len: int, batch_size: int):
    model = model_class.from_pretrained(
        pretrained_path, torch_dtype=torch.bfloat16, device_map="auto", use_flash_attention_2=True
    )
    model.resize_token_embeddings(tokenizer_len)
    if num_new_tokens > 0: # initialize new embeddings = mean old embeddings
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
        
    return compute_loss_from_ds_by_model(ds, model, batch_size)


def main(
    pretrained_path: str,  
    test_path: str = typer.Option("functionary/train/small_tests.jsonl"),
    padding_size: str = typer.Option("left"),
    max_length: int = typer.Option(4096),
    batch_size: int = typer.Option(3),
):
    """This function is used to see the difference between applying packing vs not applying packing.
    We will read the data from test_path, then load into 2 types of datasets: CustomDataset (wo packing) and PackedDataset (with packing)
    We will compute the total loss from: loss1 = (CustomDataset, MistralModel) vs loss2 = (PackedDataset, Monkey-patched MistralModel)
    We compare loss1 and loss2, if loss1 and loss2 are almost the same --> our implementation is correct
    Args:
        pretrained_path (str): _description_
        test_path (str, optional): _description_. Defaults to typer.Option("functionary/train/small_tests.jsonl").
        padding_size (str, optional): _description_. Defaults to typer.Option("left").
        max_length (int, optional): _description_. Defaults to typer.Option(4096).
    """
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path, legacy=True, model_max_length=max_length)
    tokenizer.padding_size = padding_size
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": get_additional_tokens()}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    
    with open(test_path, "r") as f:
        dt_points = [json.loads(line) for line in f]

    normal_ds = custom_datasets.CustomDataset(dt_points, tokenizer)
    print("normal dp example: ", normal_ds[0])
    packed_ds = custom_datasets.FAPackedDataset(dt_points, tokenizer)
    print("packed da example: ", packed_ds[0])
    print(f"number of data points: {len(dt_points)}, normal dataset size: {len(normal_ds)}; packed dataset size: {len(packed_ds)}")
    
    normal_loss = compute_loss_from_ds(normal_ds, pretrained_path, OriginalMistral, num_new_tokens, len(tokenizer), batch_size)
    normal_loss = normal_loss / len(dt_points)
    mk_loss = compute_loss_from_ds(packed_ds, pretrained_path, MonkeyPatchedMistral, num_new_tokens, len(tokenizer), batch_size)
    mk_loss = mk_loss / len(dt_points)
    
    diff = math.fabs(normal_loss - mk_loss)
    diff_percent = diff * 100 / max(normal_loss, mk_loss)
    print(f"normal_loss: {normal_loss:0.3f}, mk_loss={mk_loss:0.3f}, diff_percent={diff_percent:0.3f} % ")


if __name__ == "__main__":
    typer.run(main)
