import unittest
from schema import generate_schema_for_functions

class TestSchemaGenerator(unittest.TestCase):
    def test_generate_schema_for_functions(self):
        self.maxDiff = None
        functions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
            {
                "name": "get_n_day_weather_forecast",
                "description": "Get an N-day weather forecast",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                        "num_days": {
                            "type": "integer",
                            "description": "The number of days to forecast",
                        }
                    },
                    "required": ["location", "format", "num_days"]
                },
            },
        ]

        namespace = "weather"

        expected_output = '''namespace weather {

// Get the current weather
type get_current_weather  = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
// The temperature unit to use. Infer this from the users location.
format: "celsius" | "fahrenheit",
}) => any;

// Get an N-day weather forecast
type get_n_day_weather_forecast  = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
// The temperature unit to use. Infer this from the users location.
format: "celsius" | "fahrenheit",
// The number of days to forecast
num_days: number,
}) => any;

} // namespace weather'''

        actual_output = generate_schema_for_functions(functions, namespace)
        self.assertEqual(actual_output, expected_output)

if __name__ == '__main__':
    unittest.main()