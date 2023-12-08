from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

print(
    client.chat.completions.create(
        model="/workspace/functionary-7b-v1.4/",
        messages=[{"role": "user", "content": "What is the weather for Istanbul?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
    )
)
