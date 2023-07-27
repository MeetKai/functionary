from typing import Dict, Any, Optional, List
from collections import OrderedDict
from ..schema_gen import Function
import re


def generate_from_openapi_v301(specification: Dict[str, Any], 
                          namespace: str = 'plugins', 
                          description: Optional[str] = None) -> List[Function]:
    """
    Converts an OpenAPI specification to a list of Function objects.

    :param specification: An OpenAPI specification as a dictionary.
    :param namespace: The namespace for the generated TypeScript schema (default is 'plugins').
    :param description: A description for the OpenAPI specification (optional).

    :return: A list of Function objects derived from the OpenAPI specification.
    """

    function_objects = []  # List to store the created Function objects

    paths = specification.get('paths', {})
    if not paths:
        raise ValueError('The provided OpenAPI specification does not contain any paths.')

    for path, path_content in paths.items():
        for method, method_content in path_content.items():
            function_name = re.sub(r'{[^}]*}', '', path).replace('/', '_').strip('_')
            parameters = method_content.get('parameters', [])

            parameter_info = {
                'type': 'object',
                'properties': {},
                'required': []
            }

            for parameter in parameters:
                param_name = parameter.get('name')
                param_type = parameter.get('schema', {}).get('type')
                param_description = parameter.get('description', '')
                if param_name and param_type: 
                    parameter_info['properties'][param_name] = {
                        'type': param_type,
                        'description': param_description
                    }
                    if parameter.get('required'):
                        parameter_info['required'].append(param_name)

            if parameter_info['properties']:
                transformed_schema = {
                    'name': function_name,
                    'description': method_content.get('summary', ''),
                    'parameters': parameter_info,
                }

                func_object = Function(transformed_schema, namespace=namespace)
                function_objects.append(func_object)  # Add the Function object to the list

    return function_objects

def generate_from_openapi_v300(specification: Dict[str, Any], 
                          namespace: str = 'plugins', 
                          description: Optional[str] = None) -> List[Function]:
    """
    Converts an OpenAPI specification to a list of Function objects.

    :param specification: An OpenAPI specification as a dictionary.
    :param namespace: The namespace for the generated TypeScript schema (default is 'plugins').
    :param description: A description for the OpenAPI specification (optional).

    :return: A list of Function objects derived from the OpenAPI specification.
    """

    function_objects = []  # List to store the created Function objects

    paths = specification.get('paths', {})
    if not paths:
        raise ValueError('The provided OpenAPI specification does not contain any paths.')

    for path, path_content in paths.items():
        for method, method_content in path_content.items():
            function_name = re.sub(r'{[^}]*}', '', path).replace('/', '_').strip('_')
            parameters = method_content.get('requestBody', {}).get('content', {}).get('application/json', {}).get('schema', {}).get('properties', {})

            parameter_info = {
                'type': 'object',
                'properties': {},
                'required': []
            }

            for param_name, param_content in parameters.items():
                param_type = param_content.get('type')
                param_description = param_content.get('description', '')
                if param_name and param_type: 
                    parameter_info['properties'][param_name] = {
                        'type': param_type,
                        'description': param_description
                    }
                    if param_content.get('required'):
                        parameter_info['required'].append(param_name)

            if parameter_info['properties']:
                transformed_schema = {
                    'name': function_name,
                    'description': method_content.get('summary', ''),
                    'parameters': parameter_info,
                }

                func_object = Function(transformed_schema, namespace=namespace)
                function_objects.append(func_object)  # Add the Function object to the list

    return function_objects

def generate_from_openapi_v2(specification: Dict[str, Any], 
                             namespace: str = 'plugins', 
                             description: Optional[str] = None) -> List[Function]:
    """
    Converts an OpenAPI v2.0 specification to a list of Function objects.

    :param specification: An OpenAPI v2.0 specification as a dictionary.
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
                if parameter.get('in') not in ['query', 'header', 'path', 'formData', 'body']:
                    continue

                param_name = parameter.get('name')
                if param_name:
                    param_type = parameter.get('type')
                    if not param_type and parameter.get('schema'):  # If type is not directly given, it may be inside 'schema'
                        param_type = parameter.get('schema', {}).get('type')

                    if param_type:  # only append if type exists and is not empty
                        param_description = parameter.get('description', '')
                        parameter_info['properties'][param_name] = {'type': param_type, 'description': param_description}
                        if parameter.get('required'):
                            parameter_info['required'].append(param_name)

            # After all parameters have been processed, add 'type' key at the beginning
            parameter_info = OrderedDict([('type', 'object')] + list(parameter_info.items()))

            if parameter_info['properties']:
                transformed_schema = {
                    'name': function_name,
                    'description': method_content.get('summary', ''),
                    'parameters': parameter_info,
                }

                func_object = Function(transformed_schema, namespace=namespace)
                function_objects.append(func_object)  # Add the Function object to the list

    return function_objects


