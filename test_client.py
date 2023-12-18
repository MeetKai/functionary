from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

print(
    client.chat.completions.create(
        model="meetkai/functionary-7b-v2",
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Istanbul?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in Istanbul",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            # "location": {
                            #     "type": "string",
                            #     "description": "The city and state, e.g. San Francisco, CA",
                            # }
                        },
                        "required": [],
                    },
                },
            }
        ],
        tool_choice="auto",
    )
)
