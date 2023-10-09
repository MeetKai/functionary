from functionary import schema
from functionary.openai_types import Function
import utility
import sys 
import json


def get_list_funct_json():
    items =  [
        {
        "name": "fetchCaptions",
        "description": "Get the captions for a YouTube video in various languages",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "string"
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the YouTube video"
                },
                "type": {
                    "type": "string",
                    "default": "manual",
                    "enum": [
                        "auto",
                        "manual"
                    ],
                    "description": "The type of transcript to fetch ('auto' or 'manual')"
                },
                "lang": {
                    "type": "string",
                    "description": "The language code for the captions"
                },
                "itemIds": {
                    "type": "array",
                    "description": "list of ids",
                    "items": {
                        "type": "string"
                    }
                },
                "from_dt": {
                    "type": "string",
                    "format": "date-time",
                    "description": "from datetime"
                },
                "end_dt": {
                    "type": "string",
                    "format": "date-time"
                },
                "published_utc.lte": {
                    "oneOf": [
                        {
                            "format": "date-time",
                            "type": "string"
                        },
                        {
                            "format": "date",
                            "type": "string"
                        }
                    ],
                    "description": "Search for published utc values that are less than or equal to the given value"
                },
                "items_per_page": {
                    "type": "integer",
                    "maximum": 36,
                    "description": "Number of items that should be returned per page."
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of page that should be returned."
                },
                "person": {
                    "type": "object",
                    "description": "Number of page that should be returned.",
                    "properties": {
                        "name": {"type": "string", "description": "name of person"},
                        "age": {"type": "integer", "description": "age of person"},
                        "extra_info": {
                            "properties": {
                                    "school": {"type": "string", "description": "school of this person"},
                                    "job": {"type": "string", "description": "job of this person"}
                                },
                            "type": "object",
                            "description": "extra information of this person."
                        }
                    }
                }
            },
            "required": [
                "url",
                "lang"
            ]
        }
    }
    ]
    return [Function(**func) for func in items]


def get_list_functions():
    return [
        Function(name="get_car_price", description="This is get_car_price", parameters={
                "type": "object",
                "properties": {
                    "car_name": {
                        "type": "string",
                        "description": "name of the car"
                    }
                },
                "required": [
                    "car_name"
                ]
        }),
        Function(name="get_weather", description="This is get_weather", parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "default": "Hanoi",
                        "description": "city we want to know the weather"
                    },
                    "date": {
                        "type": "string",
                        "description": "weather of which date"
                    }
                },
                "required": [
                    "date"
                ]
        }),
        Function(name="book_hotel", description="This is for booking hotel", parameters={
                "type": "object",
                "properties": {
                    "hotel": {
                        "type": "string",
                        "description": "hotel to book"
                    },
                    "date": {
                        "type": "object",
                        "description": "information about date",
                        "properties": {
                            "from_date": {"type": "string", "description": "from which date"},
                            "to_date": {"type": "string", "description": "to which date"}
                        }
                    }
                },
                "required": [
                    "date"
                ]
        }),
    ]
    

def filter_out_functions(jsonl_path, save_path):
    items = utility.read_jsonl(jsonl_path)
    print("number of items: ", len(items))
    f_items = []
    function_dic = dict()
    for item in items:
        functions = item.get("functions", [])
        if len(functions) == 1 and functions[0]["name"] == "python":
                f_items.append(item)
    print("number of unique funcs: ", len(function_dic))
    utility.save_json(f_items, save_path)


def investigate_functions():
    path = "functions.json"
    items = utility.read_json(path)
    inside_fields = dict()
    field_value_dic = dict()
    tracked_fields = ["minimum", "maxLength", "maximum"]
    typeset_count = dict()
    for item in items:
        properties = item["parameters"]["properties"]
        for param_name in properties:
            param = properties[param_name]
            for key in param:
                inside_fields[key] = inside_fields.get(key, 0) + 1
                if key in tracked_fields:
                    if key not in field_value_dic:
                        field_value_dic[key] = set()
                    field_value_dic[key].add(param[key])
            param_type = param.get("type")
            if param_type == "object":
                print('---------------')
                print(json.dumps(param, ensure_ascii=False, indent=4))
    
    for key, value in sorted(inside_fields.items(), key=lambda x:- x[1]):
        print(f"{key}: {value}")
    print("---------")
    print("typeset: ")
    print(json.dumps(typeset_count, ensure_ascii=False, indent=4))
    
    print("field_value_dic: ")
    for key in field_value_dic:
        field_value_dic[key] = list(field_value_dic[key])
    print(json.dumps(field_value_dic, ensure_ascii=False, indent=4))
    

def test_schema():
    functions = get_list_functions() + get_list_funct_json()
    result = schema.generate_schema_from_functions(functions)
    print(result)


if __name__ == "__main__":
    #investigate_functions()
    filter_out_functions("2023-10-04.jsonl", "functions_1.json")
    #test_schema()

