##Unit tests for schema gen

import unittest
from unittest.mock import patch
import requests_mock
from .schema_gen import 


class TestFunctionaryUtils(unittest.TestCase):
    def setUp(self):
        self.sample_function_definition = {
            "name": "testFunction",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "description": "First parameter",
                        "type": "string"
                    }
                },
                "required": ["param1"]
            }
        }
        self.sample_openapi_spec =  {
    "openapi": "3.0.0",
    "info": {"version": "1.0.0", "title": "Test API"},
    "paths": {
        "/test": {
            "get": {
                "summary": "Test endpoint",
                "parameters": [
                    {
                        "name": "param1",
                        "in": "query",
                        "description": "A test parameter",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Successful operation",
                    },
                },
            }
        }
    },
}

        self.sample_url = 'http://example.com/openapi.json'

    @requests_mock.Mocker()
    def test_workflow(self, m):
        # Mock the response from the URL to return the sample OpenAPI spec
        m.get(self.sample_url, json=self.sample_openapi_spec)

        # Run the main workflow with the sample function definition and URL
        result = Functionary_Utils.SchemaGen()(
            functions=[self.sample_function_definition],
            plugin_urls=[self.sample_url]
        )

        # Verify the results
        self.assertIn('typescript_schema', result)
        print(result["typescript_schema"])


if __name__ == '__main__':
    unittest.main()
