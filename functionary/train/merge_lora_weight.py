import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from transformers import AutoModelForCausalLM, AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from peft import PeftModel
import torch
import typer
import transformers
import math


def merge_weight(
    save_folder: str,
    pretrained_path: str,
    checkpoint: str,
    model_max_length: int,
    prompt_template_version: str,
):
    print("save to: ", save_folder)
    print("pretrained: ", pretrained_path)
    print("checkpoint: ", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_template = get_prompt_template_by_version(prompt_template_version)
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()
    special_tokens = {
        "additional_special_tokens": prompt_template.get_additional_tokens()
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    print("number of new tokens: ", num_new_tokens)

    config = transformers.AutoConfig.from_pretrained(pretrained_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print("need to scale ...")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
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
