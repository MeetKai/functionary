from typing import List, Dict, Any, Optional
from openapi_spec_validator import validate_spec
from openapi_spec_validator.readers import read_from_filename 
from collections import OrderedDict
import collections
import requests
import yaml
import re


class Function:
    def __init__(self, definition, namespace):
        """
        Initializes a Function object with a given definition and namespace.

        :param definition: A dictionary representing the function definition.
        :param namespace: A string representing the namespace of the function.
        """
        self.definition = definition
        self.namespace = namespace
        self.name = definition['name']

    def __eq__(self, other):
        """
        Compares two Function objects for equality.

        :param other: Another Function object to compare with.

        :return: True if the definitions of the two Function objects are identical, False otherwise.
        """

        return self.definition == other.definition

    def __hash__(self):
        """
        Generates a hash value for a Function object.

        :return: An integer representing the hash value of the Function object.
        """

        return hash(str(self.definition))



class SchemaGen:
    
    @staticmethod
    def __call__(functions : list = None , plugin_urls : list = None, namespace = 'auto' ):
        """
        Please note plugins do not do anything yet, thus the argument will be ignored. expect this to be fixed in a few days when server side plugin execution is implemented
        """
        functions_list = []
        errors = []
        
        if functions is not None:
            if namespace == 'auto' :
                json_functions = SchemaGen.generate_from_func_dict(functions=functions, namespace='functions')
            else:
                json_functions = SchemaGen.generate_from_func_dict(functions=functions, namespace=namespace)
            
            for func in json_functions:

                functions_list.append(func)

        if plugin_urls is not None:
            for url in plugin_urls:
                schema = SchemaGen.get_openapi_spec_from_url(url)
                name, is_valid = SchemaGen.is_valid_openapi_spec(schema, from_url=True)
                if is_valid:
                    if namespace == 'auto' : 
                        function_objs = SchemaGen.generate_from_openapi(schema, namespace='plugins')
                        for function in function_objs:
                            functions_list.append(function)
                    else: 
                        function_objs  = SchemaGen.generate_from_openapi(schema, namespace=namespace)
                        for function in function_objs:
                            functions_list.append(function)
                else:
                    errors.append({"error_type" : "invalid openapi spec", "details" : {
                        "name" : name, "url" : url, "error" : "invalid spec"
                    }})
                
        typescript_schema = SchemaGen.generate_schemas(functions_list)

        resp = {"typescript_schema": typescript_schema}

        # Only include the 'errors' key in the response if there are any errors
        if errors:
            resp['errors'] = errors

        return resp



    @staticmethod
    def generate_from_func_dict(functions : List , namespace : str):
        """
        Generates a list of Function objects from provided function definitions.

        :param functions: A list of function definitions.
        :param namespace: The namespace for the generated TypeScript schema .

        :return: A list of Function objects.
        """

        functions_list = []
        for function in functions:
            obj = Function(function, namespace=namespace)
            functions_list.append(obj)

        return functions_list




    @staticmethod
    def generate_schemas(functions, default_namespace="functions", indent=2):
        """
        Generates TypeScript schema from a list of Function objects.

        :param functions: A list of Function objects.
        :param namespace: The namespace for the generated TypeScript schema (default is 'functions').
        :param indent: The number of spaces for indentation in the generated TypeScript code (default is 2).

        :return: A dictionary containing the generated TypeScript schema and any format errors.
        """

        """
        Generates TypeScript schema from a list of Function objects.

        :param functions: A list of Function objects.
        :param default_namespace: The default namespace for the generated TypeScript schema (default is 'functions') 
            if a function object does not have a namespace attribute.
        :param indent: The number of spaces for indentation in the generated TypeScript code (default is 2).

        :return: A dictionary containing the generated TypeScript schema and any format errors.
        """

        errors = []
        schema = ""
        unique_functions_by_namespace = collections.defaultdict(list)

        for function_obj in functions:
            if not isinstance(function_obj, Function):
                errors.append("Each function must be a Function object.")
                continue

            # No need to wrap the function dict in a Function object, it already is one
            function = function_obj.definition

            # Get the namespace from the function object, if it does not exist, use the default one
            namespace = getattr(function_obj, 'namespace', default_namespace)

            # Check if an identical function is already in the list for this namespace
            if function_obj not in unique_functions_by_namespace[namespace]:
                # If not, add it
                unique_functions_by_namespace[namespace].append(function_obj)

        # Now generate the TypeScript definitions for each namespace
        for namespace, unique_functions in unique_functions_by_namespace.items():
            schema += f"// Supported function definitions that should be called when necessary.\n"
            schema += f"namespace {namespace} {{\n\n"

            for function_obj in unique_functions:
                function = function_obj.definition  # Unwrap the Function object back into a dict


                function_name = function.get("name", None)
                if function_name is None:
                    errors.append("Function name is missing.")
                    continue

                description = function.get("description", "")
                schema += f"{' ' * indent}// {description}\n"
                schema += f"{' ' * indent}type {function_name}"

                parameters = function.get("parameters", None)
                if parameters is not None:
                    if not isinstance(parameters, dict):
                        errors.append("`parameters` must be a dictionary.")
                        continue

                    schema += f" = (_: {{\n"
                    required_params = parameters.get("required", [])
                    if not isinstance(required_params, list):
                        errors.append("`required` must be a list.")
                        continue

                    properties = parameters.get("properties", {})
                    if not isinstance(properties, dict):
                        errors.append("`properties` must be a dictionary.")
                        continue

                    for param_name, param in properties.items():
                        if not isinstance(param, dict):
                            errors.append(f"Each property of `{param_name}` must be a dictionary.")
                            continue

                        # Param Description
                        description = param.get("description")
                        if description is not None:
                            schema += f"{' ' * (indent*2)}// {description}\n"

                        # Param Name
                        schema += f"{' ' * (indent*2)}{param_name}"
                        if param_name not in required_params:
                            schema += '?'

                        # Param Type
                        param_type = param.get("type", "any")
                        if param_type == 'integer':
                            param_type = 'number'
                        if 'enum' in param:
                            enum = param.get('enum', [])
                            if not isinstance(enum, list):
                                errors.append(f"`enum` of `{param_name}` must be a list.")
                                param_type = "any"
                            else:
                                param_type = ' | '.join([f'"{v}"' for v in enum])
                        schema += f": {param_type},\n"

                    schema += f"{' ' * indent}}}) => any;\n\n"
                else:
                    # Doesn't have any parameters
                    schema += f" = () => any;\n\n"

            schema += f"}} // namespace {namespace}\n\n"

        if errors:
            return {"typescript_definitions": schema, "format_errors": errors}
        else:
            return schema



        
    @staticmethod
    def get_openapi_spec_from_url(url):
        """
        Fetches an OpenAPI specification from a provided URL.

        :param url: The URL to fetch the OpenAPI specification from.

        :return: The fetched OpenAPI specification as a dictionary.

        :raises ValueError: If the OpenAPI specification cannot be fetched or parsed from the URL.
        """

        # List of common endpoints for OpenAPI specifications
        common_endpoints = ['', '/openapi', '/swagger', '/api-docs', '/v1/api-docs']

        # File extensions to try
        file_extensions = ['.json', '.yaml']

        for endpoint in common_endpoints:
            full_url = url.rstrip('/') + endpoint
            try:
                response = requests.get(full_url)
                response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

                # Try to parse the response as JSON first
                try:
                    return response.json()
                except ValueError:  # If JSON decoding fails, try YAML
                    try:
                        return yaml.safe_load(response.text)
                    except yaml.YAMLError:
                        raise ValueError("Response is neither JSON nor YAML")

            except requests.RequestException:
                # If fetching from this endpoint fails, try adding file extensions to the endpoint
                for ext in file_extensions:
                    full_url_ext = full_url + ext
                    try:
                        response = requests.get(full_url_ext)
                        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx

                        # Try to parse the response as JSON first
                        try:
                            return response.json()
                        except ValueError:  # If JSON decoding fails, try YAML
                            try:
                                return yaml.safe_load(response.text)
                            except yaml.YAMLError:
                                raise ValueError("Response is neither JSON nor YAML")

                    except requests.RequestException:
                        # If fetching from this endpoint fails too, continue to the next endpoint
                        continue

        # If none of the endpoints returned a valid specification
        raise ValueError("No openapi spec found on the specified endpoint")
        
    def generate_from_openapi(specification: Dict[str, Any], namespace: str = 'plugins', description: Optional[str] = None) -> List[Function]:
        """
        Converts an OpenAPI specification to a list of Function objects.

        :param specification: An OpenAPI specification as a dictionary.
        :param namespace: The namespace for the generated TypeScript schema (default is 'plugins').
        :param description: A description for the OpenAPI specification (optional).

        :return: A list of Function objects derived from the OpenAPI specification.
        """


        function_objects = []  # List to store the created Function objects

        for path, path_content in specification.get('paths', {}).items():
            for method, method_content in path_content.items():
                if method not in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                    continue

                function_name = re.sub(r'{[^}]*}', '', path).replace('/', '_').strip('_')

                parameters = method_content.get('parameters', [])

                parameter_info = {'properties': {}, 'required': []}  # Start without 'type' key

                for parameter in parameters:
                    param_name = parameter.get('name')
                    if param_name:
                        param_type = parameter.get('schema', {}).get('type')
                        param_description = parameter.get('description', '')
                        if param_type and param_description:  # only append if both type and description exist and are not empty
                            parameter_info['properties'][param_name] = {'type': param_type, 'description': param_description}
                            if parameter.get('required'):
                                parameter_info['required'].append(param_name)

                # After all parameters have been processed, add 'type' key at the beginning
                parameter_info = OrderedDict([('type', 'object')] + list(parameter_info.items()))

                if parameter_info['properties'] and parameter_info['required']:
                    transformed_schema = {
                        'name': function_name,
                        'description': method_content.get('summary', ''),
                        'parameters': parameter_info,
                    }

                    func_object = Function(transformed_schema, namespace='plugins')
                    function_objects.append(func_object)  # Add the Function object to the list

        return function_objects

                


    @staticmethod
    def is_valid_openapi_spec(file, from_url = False):
        """
        Checks if an OpenAPI specification is valid.

        :param file: The OpenAPI specification to check. This can be a file path or a dictionary, depending on the value of the from_url parameter.
        :param from_url: A boolean value indicating whether the file parameter is a URL (True) or a file path (False).

        :return: A tuple containing the title of the OpenAPI specification, a boolean indicating its validity, and an error message if it is invalid.
        """
        if from_url:
            validate_spec(file)
            # directly use the 'file' parameter as 'spec_dict' because it's actually the spec dictionary when from_url=True
            spec_dict = file  
        else:
            try:
                spec_dict, spec_url = read_from_filename(file)
                validate_spec(spec_dict)
            except Exception as e:
                try:
                    title = spec_dict.get('info', {}).get('title', '')
                except Exception:
                    title = "No spec title found"
                return title, False, str(e)
        return spec_dict.get('info', {}).get('title', ''), True


