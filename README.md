# Functionary

<img align="right" width="256" height="256" src="https://github.com/musabgultekin/functionary/assets/3749407/c7a1972d-6ad7-40dc-8000-dceabe6baabd">

Functionary is a language model that can interpret and execute functions/plugins.

The model determines when to execute a function and can understand its output. It only triggers functions as needed. Function definitions are given as JSON Schema Objects, similar to OpenAI GPT function calls.

Based on [Llama 2](https://arxiv.org/abs/2307.09288).

## OpenAI compatible server

### Setup

Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) installed. Then:

```shell
pip install -r requirements.txt
python3 server.py --model "meetkai/functionary-7b-v1.1"
```

Or start blazing fast [vLLM](https://vllm.readthedocs.io/en/latest/getting_started/installation.html) server:

```shell
python3 server_vllm.py --model "meetkai/functionary-7b-v1.1" --host 0.0.0.0
```

### Server Usage

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "functionary" # We just need to set this something other than None, so it works with openai package. No API key is required.

openai.ChatCompletion.create(
    model="meetkai/functionary-7b-v1.1",
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

If you're having trouble with dependencies, and you have [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit), 
you can start your environment like this: 

```shell
sudo docker run --gpus all -it --shm-size=8g --name functionary -v ${PWD}/functionary_workspace:/workspace -p 8000:8000 nvcr.io/nvidia/pytorch:22.12-py3
```

### Call Real Python Function
To call the real python function, get the result and extract the result to respond, you can use [chatlab](https://github.com/rgbkrk/chatlab).

```python
from chatlab import Conversation
import openai
import os
openai.api_key = "functionary" # We just need to set this something other than None
os.environ['OPENAI_API_KEY'] = "functionary" # chatlab requires us to set this too
openai.api_base = "http://localhost:8000/v1"

# now provide the function with description
def get_car_price(car_name: str):
    """this function is used to get the price of the car given the name
    :param car_name: name of the car to get the price
    """
    car_price = {
        "tang": {"price": "$20000"},
        "song": {"price": "$25000"} 
    }
    for key in car_price:
        if key in car_name.lower():
            return {"price": car_price[key]}
    return {"price": "unknown"}

chat = Conversation(model="meetkai/functionary-7b-v1.1")
chat.register(get_car_price)  # register this function
chat.submit("what is the price of the car named Tang?") # submit user prompt

# print the flow
for message in chat.messages:
    role = message["role"].upper()
    if "function_call" in message:
        func_name = message["function_call"]["name"]
        func_param = message["function_call"]["arguments"]
        print(f"{role}: call function: {func_name}, arguments:{func_param}")
    else:
        content = message["content"]
        print(f"{role}: {content}")
```

The output will look like this:
```
USER: what is the price of the car named Tang?
ASSISTANT: call function: get_car_price, arguments:{
  "car_name": "Tang"
}
FUNCTION: {'price': {'price': '$20000'}}
ASSISTANT: The price of the car named Tang is $20,000.
```

# Use Cases

Here are a few examples of how you can use this function calling system:

### Travel and Hospitality - Trip Planning
The function `plan_trip(destination: string, duration: int, interests: list)` can take user input such as "I want to plan a 7-day trip to Paris with a focus on art and culture" and generate an itinerary accordingly.

<details>
  <summary>Details (click to expand)</summary>

```python
openai.ChatCompletion.create(
    model="meetkai/functionary-7b-v1.1",
    messages=[
        {"role": "user", "content": 'I want to plan a 7-day trip to Paris with a focus on art and culture'},
    ], 
    functions=[
        {
            "name": "plan_trip",
            "description": "Plan a trip based on user's interests",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination of the trip",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "The duration of the trip in days",
                    },
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The interests based on which the trip will be planned",
                    },
                },
                "required": ["destination", "duration", "interests"],
            },
        },
    ]
)
```

Response will have: 

```json
{"role": "assistant", "function_call": {"name": "plan_trip", "arguments": '{\n  "destination": "Paris",\n  "duration": 7,\n  "interests": ["art", "culture"]\n}'}}
```

Then you need to call ```plan_trip``` function with provided arguments. 
If you would like a commentary from the model, then you'll call the model again with the response from the function, the model will write necessary commentary.

</details>


### Real Estate - Property Valuation
A function like estimate_property_value(property_details: dict) could allow users to input details about a property (such as location, size, number of rooms, etc.) and receive an estimated market value.

<details>
  <summary>Details (click to expand)</summary>

```python
openai.ChatCompletion.create(
    model="meetkai/functionary-7b-v1.1",
    messages=[
        {"role": "user", "content": 'What is the estimated value of a 3-bedroom house in San Francisco with 2000 sq ft area?'},
        {"role": "assistant", "function_call": {"name": "estimate_property_value", "arguments": '{\n  "property_details": {"location": "San Francisco", "size": 2000, "rooms": 3}\n}'}},
    ], 
    functions=[
        {
            "name": "estimate_property_value",
            "description": "Estimate the market value of a property",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_details": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location of the property",
                            },
                            "size": {
                                "type": "integer",
                                "description": "The size of the property in square feet",
                            },
                            "rooms": {
                                "type": "integer",
                                "description": "The number of rooms in the property",
                            },
                        },
                        "required": ["location", "size", "rooms"],
                    },
                },
                "required": ["property_details"],
            },
        },
    ]
)
```

Response will have: 

```json
{"role": "assistant", "function_call": {"name": "plan_trip", "arguments": '{\n  "destination": "Paris",\n  "duration": 7,\n  "interests": ["art", "culture"]\n}'}}
```

Then you need to call ```plan_trip``` function with provided arguments. 
If you would like a commentary from the model, then you'll call the model again with the response from the function, the model will write necessary commentary.

</details>


### Telecommunications - Customer Support
A function `parse_customer_complaint(complaint: {issue: string, frequency: string, duration: string})` could help in extracting structured information from a complex, narrative customer complaint, identifying the core issue and potential solutions. The `complaint` object could include properties such as `issue` (the main problem), `frequency` (how often the issue occurs), and `duration` (how long the issue has been occurring).

<details>
  <summary>Details (click to expand)</summary>

```python
openai.ChatCompletion.create(
    model="meetkai/functionary-7b-v1.1",
    messages=[
        {"role": "user", "content": 'My internet has been disconnecting frequently for the past week'},
    ], 
    functions=[
        {
            "name": "parse_customer_complaint",
            "description": "Parse a customer complaint and identify the core issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "complaint": {
                        "type": "object",
                        "properties": {
                            "issue": {
                                "type": "string",
                                "description": "The main problem",
                            },
                            "frequency": {
                                "type": "string",
                                "description": "How often the issue occurs",
                            },
                            "duration": {
                                "type": "string",
                                "description": "How long the issue has been occurring",
                            },
                        },
                        "required": ["issue", "frequency", "duration"],
                    },
                },
                "required": ["complaint"],
            },
        },
    ]
)
```

Response will have:

```json
{"role": "assistant", "function_call": {"name": "parse_customer_complaint", "arguments": '{\n  "complaint": {"issue": "internet disconnecting", "frequency": "frequently", "duration": "past week"}\n}'}}
```

Then you need to call parse_customer_complaint function with provided arguments.
If you would like a commentary from the model, then you'll call the model again with the response from the function, the model will write necessary commentary.

</details>


## Training

We use standard HuggingFace Trainer. When calculating the loss, we only calculate the loss on assistant outputs and assistant function calls. Not on function responses and function definitions

We use the similar hyperparameters as its used in LLama 2 [paper](https://arxiv.org/abs/2307.09288). 
Except we use bigger weight decay (0.3 instead of 0.1) and warmup of 0.03, to reduce overfitting as we sample 2x of the function calling example conversations. But ablation study is required.

We use transformers after this [commit](https://github.com/huggingface/transformers/commit/f4eb459ef25c62c4cc9edde38052da1980977872). As it fixes OOM for FSDP training on Llama 2.

**Hyperparameters**:

- Batch size: 64
- Learning rate: 2e-5
- Epochs: 2
- Max length: 4096
- Weight decay: 0.3

More on training: [README.md](functionary/train/README.md) 

## How it Works?

We convert function definitions to a similar text like TypeScript definitions. 
Then we inject these definitions as system prompts. After that, we inject the default system prompt. 
Then we start the conversation messages. 

Here is an example prompt that will be provided to the model:
```text
system:
// Supported function definitions that should be called when necessary.
namespace weather {

// Get the current weather
type get_current_weather  = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
// The temperature unit to use. Infer this from the users location.
format: "celsius" | "fahrenheit",
}) => any;

} // namespace weather
system:
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary
user:
</s>What is the weather in Istanbul?</s>
assistant
```

The model will output:

```text
 to=weather.get_current_weather:
{"location": "Istanbul", "format": "celsius"}</s>
```

Then it will stop.

We don't change the logit probabilities to conform a certain schema, but the model itself knows how to conform. This allows us to use existing tools and caching systems with ease.

## Evaluation

### [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) leaderboard
|                       | MT-Bench |
|:----------------------|---------:|
| GPT-4                 |     8.99 |
| Claude-2              |     8.06 |
| GPT-3.5-turbo         |     7.94 | 
| Claude-1              |     7.90 |
| WizardLM-70B-v1.      |     7.71 |
|**Functionary-7B-v1.4**| **7.22** |
| WizardLM-13B-v1.2     |     7.20 |
| Vicuna-33B            |     7.12 |
| Llama-2-70b-chat      |     6.86 |
| Llama-2-13B-chat      |     6.65 |
| Vicuna-13B            |     6.57 |
| Tulu-30B              |     6.43 |
| Vicuna-7B             |     6.17 |
| Functionary-7B-v1     |     6.15 |
| Nous-Hermes-13B       |     5.51 |
| Koala-13B             |     5.35 |
| Falcon-40B-Instruct   |     5.17 |
| Alpaca-13B            |     4.53 |
| LLaMA-13B             |     2.61 |

### [Alpaca Eval](https://github.com/tatsu-lab/alpaca_eval) Leaderboard
    
|                       | Win Rate | Std Error |
|:----------------------|---------:|----------:|
| gpt4                  |     95.3 |       0.7 |
| claude                |     88.4 |       1.1 |
| chatgpt               |     86.1 |       1.2 |
| wizardlm-13b          |     75.3 |       1.5 |
| guanaco-65b           |     71.8 |       1.6 |
| vicuna-13b            |     70.4 |       1.6 |
| oasst-rlhf-llama-33b  |     66.5 |       1.7 |
|**functionary-7b-v1**  | **62.6** |   **1.7** |
| text_davinci_003      |     50.0 |       0.0 |
| falcon-40b-instruct   |     45.7 |       1.8 |
| alpaca-farm-ppo-human |     41.2 |       1.7 |
| alpaca-7b             |     26.5 |       1.5 |
| text_davinci_001      |     15.2 |       1.2 |

</details>

## Dataset

--- Work In Progress ---

Dataset preparation process consists of several steps:

1. **Function Definitions Conversion:** We begin by selecting multiple function definitions and converting them into TypeScript definitions. This approach benefits from the model's prior exposure to TypeScript tokens during the pretraining phase. [See how we do it](https://github.com/musabgultekin/functionary/blob/17a86de9b06acaedd0afab212717205c0484a218/schema.py#L54) Also see [Microsoft TypeChat](https://github.com/microsoft/TypeChat/blob/d2f2de9ca37ef9adeb108d5fc60703b72fec0a22/site/src/blog/introducing-typechat.md#just-add-types)

2. **Human Prompts Generation:** We then create human prompts that incorporate the converted TypeScript function definitions. 

3. **Function Calls Generation:** Following the generation of human prompts, we proceed to generate corresponding function calls.

4. **Function Answers Generation:** Once function calls have been generated, we derive the outputs of these function calls would produce.

5. **Function Answers Interpretation:** After procuring function answers, we generate language model answers for the function response. So the model knows how to interpret the function response.

6. **Merging and Training:** We combine all the generated elements (prompts, function calls, function answers, and their interpretations) using a custom formatting. This consolidated dataset is then used for the model's training.

*Note: Llama 2 70b is capable of doing all synthetic data generation.*

*More information about this process will be provided soon as possible.*

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

### v1

**Data Sources:**
- Same as v0.2

**Observations:**
Compared to v0.2, because the model supports 4k context sizes, its much more resilient to the longer conversations and longer function definitions. Also we switched to Llama 2.


## Roadmap

- [ ] Train [Llama 2](https://arxiv.org/abs/2307.09288) 13B model too, with 2x more data.
- [ ] OpenAPI specification based plugin support.
- [X] Fast inference server 
  - [X] [vLLM](https://github.com/vllm-project/vllm) 
  - [ ] [text-generation-inference](https://github.com/huggingface/text-generation-inference) ? See: [License Issue](https://github.com/huggingface/text-generation-inference/issues/726)
  - [ ] Streaming Support
  - [ ] function_call parameter to server
- [ ] Python function calling support (Automatic detection of type annotations and calling them automatically)
- [ ] Real world usage examples, such as creating agents.
- **Please consider opening a PR for future requests**
