import requests
from rich import print

YOUR_API_KEY = ""

headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Api-Key {YOUR_API_KEY}",
}

messages = [{"role": "user","content": "What is the weather for Istanbul?"}]
tools = [
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
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

json_data = {
    "messages": messages,
    "tools": tools,
    'temperature': 0.0,
}

response = requests.post(
    'https://model-zq8yz6dw.api.baseten.co/development/predict', headers=headers, json=json_data
)
print(response.text)