# Functionary

<img align="right" width="256" height="256" src="https://github.com/musabgultekin/functionary/assets/3749407/c7a1972d-6ad7-40dc-8000-dceabe6baabd">

Functionary is a language model that can interpret and execute functions/plugins.

The model decides when to run a function and it can interpret the output of the function call. Only calling the functions when its necessary. The function definitions are provided as JSON Schema Objects, just like OpenAI GPT function calls, and we provide a drop-in replacement server.

We don't change the logit probabilities to conform a certain schema, but the model itself knows how to conform. This allows us to use existing tools and caching systems with ease.

Its based on [LLaMA](https://arxiv.org/abs/2302.13971).

## OpenAI compatible server

### Setup

Make sure you have PyTorch installed. Then:

    pip install -r requirements.txt
    python3 server.py --model "musabgultekin/functionary-7b-v0.2"

### Server Usage

```python
import openai

openai.api_key = ""
openai.api_base = "http://localhost:8000/v1"

openai.ChatCompletion.create(
    model="musabgultekin/functionary-7b-v0.2",
    messages=[{"role": "user", "content": "What is the weather for Istanbul?"}],
    functions=[{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
            },
            "required": ["location"],
        },
    }]
)

```

## Standalone Usage: 

See: [inference.py](inference.py)

    python3 inference.py

## Training

We use standard HuggingFace Trainer

**Hyperparameters**:

- Batch size: 128
- Learning rate: 2e-5
- Epochs: 3
- Max length: 2048
- Weight decay: 0

*More info and training code will be shared soon*

## Evaluation

--- Work In Progress ---

Due to the unique nature, it requires custom evaluation suite. But we can probably evaluate with gpt-4-0613, likely with a similar approach like [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

## Dataset

--- Work In Progress ---

Dataset preparation process consists of several steps:

1. **Function Definitions Conversion:** We begin by selecting multiple function definitions and converting them into TypeScript definitions. This approach benefits from the model's prior exposure to TypeScript tokens during the pretraining phase.

2. **Human Prompts Generation:** We then create human prompts that incorporate the converted TypeScript function definitions.

3. **Function Calls Generation:** Following the generation of human prompts, we proceed to generate corresponding function calls.

4. **Function Answers Generation:** Once function calls have been generated, we derive the outputs of these function calls would produce.

5. **Function Answers Interpretation:** After procuring function answers, we generate language model answers for the function response. So the model knows how to interpret the function response.

6. **Merging and Training:** We combine all the generated elements (prompts, function calls, function answers, and their interpretations) using a custom formatting. This consolidated dataset is then used for the model's training.

More information about this process will be provided soon as possible.


### v0.1 

**Data Sources:** 
- [ShareGPT 34K](https://huggingface.co/datasets/ehartford/wizard_vicuna_70k_unfiltered/blob/cfe3f5810110d4d763665c070b4a966fda43e5c5/wizard_vicuna_dataset_unfiltered.json)
- Synthetic function calling dataset (2.7k examples)

**Observations:**
This version showed limitations in handling multi-prompt conversations, likely due to the absence of multiple instructions in the function calling dataset. Also hallucinations are common, we likely need more conversation data.

### v0.2

**Data Sources:** 
- [ShareGPT 53K](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/bcd32a724d8460ebe14e1d05b0195e30e9a46cb1/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json)
- Synthetic function calling dataset (3.5k examples). Sampled 2 times.

**Observations:**
Compared to v0.1, It is capable of handling multi-turn conversations and hallucinations are reduced. I want to increase the token size to 4k with Llama 2, that should handle multi-turn conversations better.


## Roadmap

- [ ] If I can save more money, I'll train [Llama 2](https://arxiv.org/abs/2307.09288) 7B and 13B model too, with 2x more data.
- [ ] OpenAPI specification based plugin support.
- [ ] Fast inference server ([vLLM](https://github.com/vllm-project/vllm) or [text-generation-inference](https://github.com/huggingface/text-generation-inference))
- [ ] Python function calling support (Automatic detection of type annotations and calling them automatically)
- ...
