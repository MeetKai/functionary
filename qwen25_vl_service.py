from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import logging
from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
import uvicorn
import time
from functionary.prompt_template import get_prompt_template_by_version

logger = logging.getLogger(__name__)

# default: Load the model on the available device(s)
model_id = "Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
prompt_template = get_prompt_template_by_version("qwen2.5")

# default processor
processor = AutoProcessor.from_pretrained(model_id)


def convert_img_content_to_text(messages):
    result = []
    for message in messages:
        if message["role"] == "user" and type(message["content"]) == list:
            # handle this
            merged_text = ""
            for item in message["content"]:
                if item["type"] == "image":
                    merged_text += "<|vision_start|><|image_pad|><|vision_end|>"
                elif item["type"] == "text":
                    merged_text += item["text"]
            result.append({"role": "user", "content": merged_text})
        else:
            result.append(message)
    return result


def parse_llm_output(llm_output):
    """
    This function parse the llm_output of format: [text_content]<tool_call>[tool_call_content]</tool_call>...<tool_call>[tool_call_content]</tool_call>
    """
    """
    Parse LLM output into text content and tool calls.
    
    Args:
        llm_output (str): Raw output from LLM containing text and tool calls
        
    Returns:
        tuple: (text_content, list of tool call contents)
    """
    text_content = ""
    tool_call_strs = []

    # Split on tool call tags
    parts = llm_output.split("<tool_call>")

    if len(parts) > 0:
        # First part is the text content
        text_content = parts[0].strip()

        # Process remaining parts as tool calls
        for part in parts[1:]:
            if "</tool_call>" in part:
                tool_call = part.split("</tool_call>")[0].strip()
                if tool_call:
                    tool_call_strs.append(tool_call)
    tool_calls = []
    for tool_call_str in tool_call_strs:
        tool_calls.append(json.loads(tool_call_str))

    return {
        "role": "assistant",
        "content": text_content if len(text_content) > 0 else None,
        "tool_calls": tool_calls,
    }


def generate_response(messages, tools):
    text = prompt_template.get_prompt_from_messages(
        messages, tools, add_generation_prompt=True
    )

    print("--------------------------------TEXT--------------------------------")
    print(text)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.001)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]
    # print(output_text)


def test():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    generate_response(messages)


class ChatCompletionRequest(BaseModel):
    messages: list
    model: Optional[str] = "deepseek-reasoner"
    tools: Optional[list] = None
    n: Optional[int] = 3
    stream: Optional[bool] = False  # default to False


app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests - non-streaming returns full response at once, streaming yields partial chunks."""

    # create_prompt_func, parse_func = musab_create_final_messages, parse_llm_output
    result = generate_response(request.messages, request.tools)
    print("--------------------------------LLM OUTPUT--------------------------------")
    print(result)
    assistant_response = prompt_template.parse_assistant_response(result)
    choices = [{"index": 0, "message": assistant_response, "finish_reason": "stop"}]
    created_time = int(time.time())
    response = {
        "id": "1",
        "created": created_time,
        "mode": model_id,
        "choices": choices,
        "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": 0},
    }
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
