from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

print(
    client.chat.completions.create(
        model="meetkai/functionary-7b-v1.4",
        messages=[
            {"role": "user", "content": "What is the weather for Istanbul at 2pm?"}
        ],
        functions=[
            {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "time": {"type": "string", "description": "The time, e.g. 2pm"},
                    },
                    "required": ["location", "time"],
                },
            }
        ],
    )
)
