from transformers import LlamaTokenizerFast
from functionary.train import custom_datasets
from functionary.prompt import get_additional_tokens
from functionary.train.llama_attention_mask_monkey_patch import LlamaForCausalLM
import torch
import copy
import typer
import math


def create_data_points():
    return [
        {
            "messages": [
                {"role": "user", "content": "hi, how are you"},
                {"role": "assistant", "content": "I am fine, thank you and you?"},
                {"role": "user", "content": "Oh I am good, where do you live now"},
                {"role": "assistant", "content": "I live in Hanoi"}
            ],
            "functions": []
        },
        {
            "messages": [
                {"role": "user", "content": "this is a test"},
                {"role": "assistant", "content": "Oh I know that"},
                {"role": "user", "content": "you are smart"},
                {"role": "assistant", "content": "yes I am "}
            ],
            "functions": []
        }
    ]


def prepare_input_dic(input_dic, device):
    result = copy.deepcopy(input_dic)
    for key in result:
        result[key] = torch.unsqueeze(input_dic[key], 0)
        result[key] = result[key].to(device)
    result["return_dict"] = True
    result["loss_reduction"] = "sum"
    return result


def compute_loss_from_ds(ds, model, device):
    total_loss = 0
    for i in range(len(ds)):
        input_dic = ds[i]
        input_dic = prepare_input_dic(input_dic, device)
        with torch.no_grad():
            loss = model.forward(**input_dic).loss.item()
            total_loss += loss
    return total_loss


def main(pretrained_path: str, device: str = typer.Option("cuda:0")):
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path, legacy=True, model_max_length=300)
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": get_additional_tokens()}
    tokenizer.add_special_tokens(special_tokens)
    
    model = LlamaForCausalLM.from_pretrained(pretrained_path, torch_dtype=torch.bfloat16, device_map=device, use_flash_attention_2=False)
    model.resize_token_embeddings(len(tokenizer))
    
    
    dt_points = create_data_points()
    normal_ds = custom_datasets.CustomDataset(dt_points, tokenizer)
    packed_ds = custom_datasets.DirectPackedDataset(dt_points, tokenizer)
    assert len(packed_ds) == 1
    assert len(normal_ds) == 2
    
    model.eval()
    normal_loss = compute_loss_from_ds(normal_ds, model, device)
    mk_loss = compute_loss_from_ds(packed_ds, model, device)
    diff = math.fabs(normal_loss - mk_loss)
    diff_percent = diff * 100 / max(normal_loss, mk_loss)
    print(f"normal_loss: {normal_loss}, mk_loss={mk_loss}, diff_percent={diff_percent}")
    

if __name__ == "__main__":
    typer.run(main)