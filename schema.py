def generate_schema_from_functions(functions, namespace="functions"):
    """
    Convert functions array to a schema that language models can understand.
    """

    schema = f"// Supported function definitions that should be called when necessary.\n"
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
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