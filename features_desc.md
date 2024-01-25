# Functionary Features

## Single Function Call
The model's ability to execute a single function call at a time. This is useful when only one specific task or query needs to be addressed. 
<details>
  <summary>Example Single Function Call (click to expand)</summary>
  
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

client.chat.completions.create(
    model="meetkai/functionary-small-v2.2",
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
    tool_choice="auto"
)
```
The output will look like this:

```python
ChatCompletionMessage(
    content=None,
    role='assistant',
    tool_calls=[ToolCall(
        arguments='{"location": "Istanbul, Turkey"}',
        name='get_current_weather'
    )],
    function_call=None,
    tool_call_id=None,
    name=None
)
```
</details>

## Parallel Function Calls
Parallel function calling is the model's ability to perform multiple function calls together, allowing the effects and results of these function calls to be resolved in parallel. This is especially useful if functions take a long time, and reduce round trips with the API. For example, the model may call functions to get the weather in 3 different locations at the same time, which will result in a message with 3 function calls in the tool_calls array, each with an id. To respond to these function calls, add 3 new messages to the conversation, each containing the result of one function call, with a tool_call_id referencing the id from tool_calls.

<details>
  <summary>Example invoking multiple function calls in one response (click to expand)</summary>
  
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

client.chat.completions.create(
    model="meetkai/functionary-small-v2.2",
    messages=[{"role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}
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
    tool_choice="auto"
)
```
  The output will look like this:
```python
  ChatCompletionMessage(
    content=None,
    role='assistant',
    tool_calls=[ToolCall(arguments='{"location": "San Francisco, CA"}', name='get_current_weather'),
    ToolCall(arguments='{"location": "Tokyo"}',name='get_current_weather'),
    ToolCall(arguments='{"location": "Paris"}',name='get_current_weather')],
    function_call=None,
    tool_call_id=None,
    name=None
)

```
</details>

## Following Up on Missing Function Arguments
The function can identify when a function call is missing necessary arguments and subsequently ask the user for the required information. This prevents the model from making incorrect assumptions or "hallucinating" data, ensuring that the responses are accurate and based on complete information.

<details>
  <summary>Example Following Up on Missing Function Arguments (click to expand)</summary>
  
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

client.chat.completions.create(
    model="meetkai/functionary-small-v2.2",
    messages=[{"role": "user",
            "content": "What's the weather now?"}
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
    tool_choice="auto"
)
```
  The output will look like this:
```python
  ChatCompletionMessage(
    content="To provide you with the current weather, I need to know the specific location you're interested in. Could you please provide me with the name of the city and state or country you want to check the weather for?",
    role='assistant',
    tool_calls=[],
    function_call=None,
    tool_call_id=None,
    name=None
)

```
</details>

## Multi-turn
The model supports multi-turn conversations. This means the model can have conversations that go back and forth many times, like a real conversation. It maintains the context and continuity of the conversation, allowing for more complex and detailed discussions. 


## Generate Model Responses Grounded in Tools Execution Results
The functionary model can analyze tool outputs and generate model responses grounded in the outputs.

<details>
  <summary>Example Generate Model Responses Grounded in Tools Execution Results (click to expand)</summary>

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="functionary")

response = client.chat.completions.create(
    model="meetkai/functionary-small-v2.2",
    messages=[{"role": "user", "content": "Which city currently experiences worse weather conditions: Tokyo, Singapore, or Jakarta?"},
              {"role": "assistant", "content": None, "tool_calls": [{'type': 'function', "function": {"id": "call_x72mqmPz12m3dBEK7g80oYVv", "name": "get_current_weather", "arguments": '{"location": "Tokyo, Japan"}'}},
               {'type': "function", "function": {"id": "call_w5XBQqBXsKGp46Oq1EKsOIxg", "name": "get_current_weather", "arguments": '{"location": "Singapore"}'}},
               {'type': "function", "function": {"id": "call_gaisByIGJegajEfW4SGU8R8p", "name": "get_current_weather", "arguments": '{"location": "Jakarta"}'}}]},
              { "tool_call_id": "call_x72mqmPz12m3dBEK7g80oYVv","role": "tool", "name": "get_current_weather", "content": '{ "location": "Tokyo, Japan", "temperature": "15°C", "weather_condition": "Rainy", "humidity": "80%", "wind_speed": "10 km/h" }' },
              { "tool_call_id": "call_w5XBQqBXsKGp46Oq1EKsOIxg", "role": "tool", "name": "get_current_weather", "content": '{ "location": "Singapore", "temperature": "30°C", "weather_condition": "Sunny", "humidity": "70%", "wind_speed": "15 km/h" }' },
              { "tool_call_id": "call_gaisByIGJegajEfW4SGU8R8p", "role": "tool", "name": "get_current_weather", "content": '{ "location": "Jakarta", "temperature": "28°C", "weather_condition": "Cloudy", "humidity": "85%", "wind_speed": "5 km/h" }' }
    ],
    tools=[],
    tool_choice="auto"
)

```
The output will look like this:

```python
ChatCompletionMessage(
    content="Based on the current weather conditions:\n\n- Tokyo, Japan is experiencing rainy conditions with a temperature of 15°C, humidity of 80%, and a wind speed of 10 km/h.\n- Singapore is currently sunny with a temperature of 30°C, humidity of 70%, and a wind speed of 15 km/h.\n- Jakarta is cloudy with a temperature of 28°C, humidity of 85%, and a wind speed of 5 km/h.\n\nComparing these conditions, Tokyo seems to be experiencing the worse weather conditions, with rainy weather, a lower temperature, and a higher humidity compared to the other two cities. However, Jakarta has a higher wind speed, which can affect comfort levels, and Singapore is sunny and warm, which might be more pleasant for outdoor activities.",
    role='assistant',
    tool_calls=[],
    function_call=None,
    tool_call_id=None,
    name=None
)
```
  
</details>


## Chit-Chat
The functionary model not only supports function calls but the model also supports casual conversation, often referred to as chit-chat.
