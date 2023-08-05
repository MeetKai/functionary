from typing import Any, Dict, List

from functionary.openai_types import Function


def generate_schema_from_functions(functions: List[Function], namespace="functions"):
    """
    Convert functions array to a schema that language models can understand.
    """

    schema = f"// Supported function definitions that should be called when necessary.\n"
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
        # Convert Function object to dict, if necessary
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
            schema += f" = (_: {{\n"
            required_params = parameters.get("required", [])
            for param_name, param in parameters.get("properties", {}).items():
                # Param Description
                description = param.get("description")
                if description is not None:
                    schema += f"// {description}\n"

                # Param Name
                schema += f"{param_name}"
                if param_name not in required_params:
                    schema += '?'

                # Param Type
                param_type = param.get("type", "any")
                if param_type == 'integer':
                    param_type = 'number'
                if 'enum' in param:
                    param_type = ' | '.join([f'"{v}"' for v in param['enum']])
                schema += f": {param_type},\n"

            schema += f"}}) => any;\n\n"
        else:
            # Doesnt have any parameters
            schema += f" = () => any;\n\n"

    schema += f"}} // namespace {namespace}"

    return schema


def generate_schema_from_openapi(specification: Dict[str, Any], description: str, namespace: str) -> str:
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

    description_clean = description.replace('\n', '')
    
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

            if ("requestBody" in method_info) or (method_info.get("parameters") is not None):
                schema += f"  = (_: {{\n"
                # Body
                if "requestBody" in method_info:
                    body_schema = method_info.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
                    for param_name, param in body_schema.get("properties", {}).items():
                        
                        # Param Description
                        description = param.get("description")
                        if description is not None:
                            schema += f"// {description}\n"
                        
                        # Param Name
                        schema += f"{param_name}"
                        if (not param.get('required', False)) or (param.get("nullable", False)) or (param_name in body_schema.get("required", [])):
                            schema += '?'
                        
                        # Param Type
                        param_type = param.get("type", "any")
                        if param_type == 'integer':
                            param_type = 'number'
                        if 'enum' in param:
                            param_type = ' | '.join([f'"{v}"' for v in param['enum']])
                        schema += f": {param_type},\n"

                # URL
                for param in method_info.get("parameters", []):
                    # Param Description
                    if description := param.get("description"):
                        schema += f"// {description}\n"
                    
                    # Param Name
                    schema += f"{param['name']}"
                    if (not param.get('required', False)) or (param.get("nullable", False)):
                        schema += '?'
                    if param.get("schema")is None:
                        continue
                    # Param Type
                    param_type = param['schema'].get("type", "any")
                    if param_type == 'integer':
                        param_type = 'number'
                    if 'enum' in param['schema']:
                        param_type = ' | '.join([f'"{v}"' for v in param['schema']['enum']])
                    schema += f": {param_type},\n"

                schema += f"}}) => any;\n\n"
            else:
                # Doesnt have any parameters
                schema += f" = () => any;\n\n"


    schema += f"}} // namespace {namespace}"

    return schema