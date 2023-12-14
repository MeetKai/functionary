from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

total_count = 0
for i in range(30):
    output = client.chat.completions.create(
        model="meetkai/functionary-7b-v2",
        messages=[
            {
                "role": "user",
                "content": "What is the weather and time in Istanbul? How about just the weather in Singapore?",
            }
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
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_weather_and_time",
                "description": "Get the weather and time",
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
            },
        ],
        tool_choice="auto",
    )

    get_weather_count = 0
    for tool_call in output.choices[0].message.tool_calls:
        func = tool_call.function
        if func.name == "get_weather":
            get_weather_count += 1
    if get_weather_count == 2:
        total_count += 1
    print(f"{i} / 30: {get_weather_count}")

print(f"Total wrong data points: {total_count} / 30")
