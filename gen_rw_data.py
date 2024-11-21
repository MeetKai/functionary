from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_from_tokenizer
import requests
import json
import typer
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1")

model_name = "meetkai/functionary-medium-v3.3-epoch2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt_template = get_prompt_template_from_tokenizer(tokenizer)


def get_result_from_prompt(prompt, temperature, stop):
    endpoint = "http://127.0.0.1:8000/v1/prompt"
    data = {"prompt": prompt, "temperature": temperature, "stop": stop, "gen_num": 1}
    response = requests.post(endpoint, json=data)
    result = json.loads(response.text)
    return result["results"][0]


def get_response_with_reasoning(messages, tools, reasoning_temperature: float = 1.2):    
    prompt_str = prompt_template.get_prompt_from_messages(
        messages,
        tools_or_functions=tools,
        bos_token="",
        add_generation_prompt=True,
    )
    
    reasoning = get_result_from_prompt(
        prompt_str, reasoning_temperature, stop=["</reasoning>"]
    )
    prompt_str += reasoning
    action_text = get_result_from_prompt(
        prompt_str, 0.0001, prompt_template.get_stop_tokens_for_generation()
    )
    full_gen = reasoning + action_text
    result = prompt_template.parse_assistant_response(full_gen)
    return result


def get_full_result(messages, tools):
    response = client.chat.completions.create(
        model="meetkai/functionary-medium-v3.3-epoch2",  # "meetkai/functionary-medium-v3.3-epoch2",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
        stream=False,
        max_tokens=1024,
    )
    return response.choices[0].message.dict(exclude_none=True)


def main(path: str, iter_num: int = 3, temperature: float = 1.2):
    with open(path, "r") as f:
        inputs = json.loads(f.read())
    tools = []
    for tool in inputs["tools"]:
        if "type" not in tool:
            tools.append({"type": "function", "function": tool})
        else:
            tools.append(tool)

    for i in range(iter_num):
        print(f"-------------------iter:{i}, temp={temperature}---------------")
        result = get_response_with_reasoning(inputs["messages"], tools, temperature)
        print(json.dumps(result, ensure_ascii=False, indent=4))

    print("---------------result with temp=0--------------")
    result = get_response_with_reasoning(inputs["messages"], tools, 0.00001)
    print(json.dumps(result, ensure_ascii=False, indent=4))
    print("------------FULL_RESULT-------")
    full = get_full_result(inputs["messages"], tools)
    print("---------END2END RESULT----------")
    print(json.dumps(full, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    typer.run(main)
