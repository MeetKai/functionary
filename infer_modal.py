from openai import OpenAI
from rich import print

client = OpenAI(base_url="https://jeffreymeetkai--functionary-vllm-fastapi-app.modal.run/v1", api_key="functionary")

output = client.chat.completions.create(
    model="meetkai/functionary-medium-v3.2",
    messages=[{"role": "user",
            "content": "What is the weather for Istanbul?"}
    ],
    tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }],
    tool_choice="auto",
    temperature=0.0,
)

print(output)