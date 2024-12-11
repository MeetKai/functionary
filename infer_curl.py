import time

import requests

data = {
    "model": "meetkai/functionary-small-v2.4",  # model name here is the value of argument "--model" in deploying: server_vllm.py or server.py
    "messages": [{"role": "user", "content": "What is the weather for Istanbul?"}],
    # "tools": [  # For functionary-7b-v2 we use "tools"; for functionary-7b-v1.4 we use "functions" = [{"name": "get_current_weather", "description":..., "parameters": ....}]
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_current_weather",
    #             "description": "Get the current weather",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "location": {
    #                         "type": "string",
    #                         "description": "The city and state, e.g. San Francisco, CA",
    #                     }
    #                 },
    #                 "required": ["location"],
    #             },
    #         },
    #     }
    # ],
    # "tool_choice": "auto",
    "function": [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ],
}

start_time = time.time()
response = requests.post(
    "https://jeffreymeetkai-dev--functionary-vllm-fastapi-app-dev.modal.run/v1/chat/completions",
    json=data,
    headers={"Content-Type": "application/json", "Authorization": "Bearer xxxx"},
)
duration = time.time() - start_time

# Print the response text
print(response.text)
print("Duration:", duration)
