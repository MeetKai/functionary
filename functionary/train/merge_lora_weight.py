import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from transformers import AutoModelForCausalLM, LlamaTokenizer
from functionary.prompt import EndToken
from peft import PeftModel
import torch
import typer


def merge_weight(save_folder: str, pretrained_path: str, checkpoint: str):
    tokenizer = LlamaTokenizer.from_pretrained(pretrained_path, legacy=True)
    special_tokens = {"additional_special_tokens": [e.value for e in EndToken]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        device_map="auto",
        trust_remote_code=True,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
    )
    print("model = ", model)
    model.resize_token_embeddings(len(tokenizer))

    lora_model = PeftModel.from_pretrained(model, checkpoint, torch_dtype=torch.float16)
    lora_model = lora_model.merge_and_unload()
    lora_model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)
    print("final lora model: ", lora_model)


if __name__ == "__main__":
    typer.run(merge_weight)
