import pdb
from copy import deepcopy
from typing import Any, Dict, List, Optional

import jsonref
import requests
import yaml

from functionary.openai_types import Function


def normalize_data_type(param_type: str) -> str:
    if param_type == "integer" or param_type == "float":
        return "number"
    return param_type


def get_param_type(param: Dict) -> str:    
    param_type = "any"
    if "type" in param:
        param_type = param["type"]
        if param_type == "array":  # handle if param is an array 
            item_type = None
            if "items" in param and "type" in param["items"]:
                item_type = param["items"]["type"]
            if item_type is not None:
                param_type = f"Array<{item_type}>"  # For examle, Array<string>
            else:
                param_type = f"Array"
    else:
        if "oneOf" in param:
            one_of_types = []
            for item in param["oneOf"]:
                if "type" in item:
                    one_of_types.append(normalize_data_type(item["type"]))
            one_of_types = list(set(one_of_types))
            param_type = " | ".join(one_of_types)
    return normalize_data_type(param_type)


def get_format_param(param: Dict) -> Optional[str]:
    if "format" in param:
        return param["format"]
    if "oneOf" in param:
        formats = []
        for item in param["oneOf"]:
            if "format" in item:
                formats.append(item["format"])
        if len(formats) > 0:
            return " or ".join(formats)
    return None


def get_param_info(param: Dict) -> Optional[str]:
    param_type = param.get("type", "any")
    info_list = []
    if "description" in param:
        info_list.append(param["description"])
        
    if "default" in param:
        default_value = param["default"]
        if param_type == "string":
            default_value = f'"{default_value}"'  # if string --> add ""
        info_list.append(f"If not specified, use {default_value} as default.")
        
    format_param = get_format_param(param)
    if format_param is not None:
        info_list.append("The format is: " + format_param)
    
    for field, field_name in [("maximum", "maximum"), ("minimum", "minimum"), ("maxLength", "maximum length"), ("minLength", "minimum length")]:
        if field in param:
            info_list.append(f"The {field_name} value is: " + str(param[field]) + ".")
        
    if len(info_list) > 0:
        return "// " + " ".join(info_list)
    return None


def get_parameter_typescript(properties, required_params, depth=0):
    params = []
    for param_name, param in properties.items():
        # Param Description
        comment_info = get_param_info(param)
        # Param Name declaration
        param_declaration = f"{param_name}"
        if param_name not in required_params:
            param_declaration += "?"
        
        param_type = get_param_type(param)
        if param_type == "object" and "properties" in param:
            child_properties = param["properties"]
            child_required_params = param.get("required", [])
            description = param.get("description", "")
            child_desc = get_parameter_typescript(child_properties, child_required_params, depth + 1)
            if depth == 0:
                comment_info = f"/*\n {description} This is an object with the following fields:\n{child_desc}\n*/"
            else:
                comment_info = f"// {description} This is an object with the following fields:\n{child_desc}"
            param_declaration += ": object"
        else:
            if "enum" in param:
                param_type = " | ".join([f'"{v}"' for v in param["enum"]])
            param_declaration += f": {param_type}"
        params.append((comment_info, param_declaration))
    result = ""
    for comment_info, param_declaration in params:
        offset = ""
        if depth > 1:
            offset = "".join(["    " for _ in range(depth)])
        if comment_info is not None:
            if depth == 0:
                result += f"{offset}{comment_info}\n{offset}{param_declaration}\n"
            else:
                result += f"{offset}{param_declaration}  {comment_info}\n"
        else:
            result += f"{offset}{param_declaration}\n"
    return result
        

def generate_schema_from_functions(functions: List[Function], namespace="functions"):
    """
    Convert functions schema to a schema that language models can understand.
    """

    schema = "// Supported function definitions that should be called when necessary.\n"
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
        # Convert a Function object to dict, if necessary
        if not isinstance(function, dict):
            function = function.dict()
        function_name = function.get("name", None)
        if function_name is None:
            continue

        description = function.get("description", "")
        schema += f"// {description}\n"
        schema += f"type {function_name}"

        parameters = function.get("parameters", None)
        if parameters is not None:
            schema += " = (_: {\n"
            required_params = parameters.get("required", [])
            argument_info = get_parameter_typescript(parameters.get("properties"), required_params, 0)
            # for param_name, param in parameters.get("properties", {}).items():
            #     # Param Description
            #     comment_info = get_param_info(param)
            #     if comment_info is not None:
            #         schema += comment_info
            #     # Param Name
            #     schema += f"{param_name}"
            #     if param_name not in required_params:
            #         schema += "?"
                
            #     param_type = get_param_type(param)
            #     if "enum" in param:
            #         param_type = " | ".join([f'"{v}"' for v in param["enum"]])
            #     schema += f": {param_type},\n"
            schema += argument_info
            schema += "}) => any;\n\n"
        else:
            # Doesn't have any parameters
            schema += " = () => any;\n\n"

    schema += f"}} // namespace {namespace}"

    return schema


def generate_schema_from_openapi(
    specification: Dict[str, Any], description: str, namespace: str
) -> str:
    """
    Convert OpenAPI specification object to a schema that language models can understand.

    Input:
    specification: can be obtained by json.loads of any OpanAPI json spec, or yaml.safe_load for yaml OpenAPI specs

    Example output:

    // General Description
    namespace functions {

    // Simple GET endpoint
    type getEndpoint = (_: {
    // This is a string parameter
    param_string: string,
    param_integer: number,
    param_boolean?: boolean,
    param_enum: "value1" | "value2" | "value3",
    }) => any;

    } // namespace functions
    """

    description_clean = description.replace("\n", "")

    schema = f"// {description_clean}\n"
    schema += f"namespace {namespace} {{\n\n"

    for path_name, paths in specification.get("paths", {}).items():
        for method_name, method_info in paths.items():
            operationId = method_info.get("operationId", None)
            if operationId is None:
                continue
            description = method_info.get("description", method_info.get("summary", ""))
            schema += f"// {description}\n"
            schema += f"type {operationId}"

            if ("requestBody" in method_info) or (
                method_info.get("parameters") is not None
            ):
                schema += f"  = (_: {{\n"
                # Body
                if "requestBody" in method_info:
                    try:
                        body_schema = (
                            method_info.get("requestBody", {})
                            .get("content", {})
                            .get("application/json", {})
                            .get("schema", {})
                        )
                    except AttributeError:
                        body_schema = {}
                    for param_name, param in body_schema.get("properties", {}).items():
                        # Param Description
                        description = param.get("description")
                        if description is not None:
                            schema += f"// {description}\n"

                        # Param Name
                        schema += f"{param_name}"
                        if (
                            (not param.get("required", False))
                            or (param.get("nullable", False))
                            or (param_name in body_schema.get("required", []))
                        ):
                            schema += "?"

                        # Param Type
                        param_type = param.get("type", "any")
                        if param_type == "integer":
                            param_type = "number"
                        if "enum" in param:
                            param_type = " | ".join([f'"{v}"' for v in param["enum"]])
                        schema += f": {param_type},\n"

                # URL
                for param in method_info.get("parameters", []):
                    # Param Description
                    if description := param.get("description"):
                        schema += f"// {description}\n"

                    # Param Name
                    schema += f"{param['name']}"
                    if (not param.get("required", False)) or (
                        param.get("nullable", False)
                    ):
                        schema += "?"
                    if param.get("schema") is None:
                        continue
                    # Param Type
                    param_type = param["schema"].get("type", "any")
                    if param_type == "integer":
                        param_type = "number"
                    if "enum" in param["schema"]:
                        param_type = " | ".join(
                            [f'"{v}"' for v in param["schema"]["enum"]]
                        )
                    schema += f": {param_type},\n"

                schema += f"}}) => any;\n\n"
            else:
                # Doesn't have any parameters
                schema += f" = () => any;\n\n"

    schema += f"}} // namespace {namespace}"

    return schema


def generate_specification_from_openapi_url(
    openapi_url: str, proxies: dict = None
) -> str:
    # Make Request
    headers = {"Accept": "application/x-yaml, text/yaml, text/x-yaml, application/json"}
    response = requests.get(
        openapi_url, verify=False, headers=headers, timeout=60, proxies=proxies
    )

    if response.status_code == 200:
        # Trust content-type first
        if response.headers.get("Content-Type") is not None:
            if "application/json" in response.headers.get("Content-Type"):
                specification = response.json()
            else:
                specification = yaml.safe_load(response.text)
        elif response.url.endswith(".json"):
            specification = response.json()
        else:
            specification = yaml.safe_load(response.text)
        # Resolve references
        specification = deepcopy(jsonref.JsonRef.replace_refs(specification))
        return specification
