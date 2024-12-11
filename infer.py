from openai import OpenAI
from rich import print

# client = OpenAI(base_url="http://158.101.19.41:8000/v1", api_key="functionary")
# client = OpenAI(
#     base_url="https://txs8ljjcm7dnku-8000.proxy.runpod.net/v1", api_key="functionary"
# )
client = OpenAI(
    base_url="https://modal.com/apps/jeffreymeetkai/dev/ap-9UaUfhPchYB22rJfiYIIgU/v1",
    api_key="functionary",
)

response = client.chat.completions.create(
    model="meetkai/functionary-small-lora-2024-10-29",
    messages=[{"role": "user", "content": "What is the weather for Istanbul?"}],
    tools=[
        {
            "type": "function",
            "function": {
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
            },
        }
    ],
    tool_choice="auto",
    temperature=0.0,
)

print(response.choices[0])
