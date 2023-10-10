import unittest
from functionary.schema import generate_schema_from_functions


class TestSchemaGenerator(unittest.TestCase):
    def test_generate_schema_for_functions(self):
        self.maxDiff = None
        functions = [
            {
                "name": "test_function",
                "description": "This is a test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "string", "description": "description of param 2"},
                        "param3": {
                            "type": "string",
                            "default": "option1",
                            "enum": ["option1", "option2"],
                            "description": "description of param 3",
                        },
                        "param4": {"type": "array", "description": "list of ids", "items": {"type": "string"}},
                        "param5": {"type": "string", "format": "date-time", "description": "from datetime"},
                        "param6": {"type": "string", "format": "date-time"},
                        "param7": {
                            "oneOf": [{"format": "date-time", "type": "string"}, {"format": "date", "type": "string"}],
                            "description": "Description of param 7",
                        },
                        "param8": {"type": "integer", "maximum": 36, "description": "description of param8"},
                        "param9": {"type": "integer", "minimum": 1, "description": "description of param 9"},
                        "person": {
                            "type": "object",
                            "description": "Number of page that should be returned.",
                            "properties": {
                                "name": {"type": "string", "description": "name of person"},
                                "age": {"type": "integer", "description": "age of person"},
                                "extra_info": {
                                    "properties": {
                                        "school": {"type": "string", "description": "school of this person"},
                                        "job": {
                                            "type": "object",
                                            "description": "job of this person",
                                            "properties": {
                                                "salary": {"type": "number", "description": "salary per month"},
                                                "title": {"type": "string", "description": "position in company"},
                                                "full_time": {
                                                    "type": "boolean",
                                                    "description": "is this person full-time or not",
                                                    "default": True,
                                                },
                                            },
                                            "required": ["salary", "title"],
                                        },
                                    },
                                    "type": "object",
                                    "description": "extra information of this person.",
                                    "required": ["job"],
                                },
                            },
                        },
                        "param10": {
                            "type": "array",
                            "description": "description of param 10",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "search": {"type": "string", "description": "this is search param"},
                                    "category": {"type": "string"},
                                },
                            },
                        },
                        "param11": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Description of param 11",
                        },
                        "param12": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "bungalow",
                                    "detached",
                                    "flat",
                                    "land",
                                    "park home",
                                    "semi-detached",
                                    "terraced",
                                ],
                            },
                        },
                        "param13": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
                    },
                    "required": ["param1", "param2"],
                },
            }
        ]

        expected_output = """// Supported function definitions that should be called when necessary.
namespace functions {

// This is a test function
type test_function = (_: {
param1: string
// description of param 2.
param2: string
// description of param 3. Default value="option1".
param3?: "option1" | "option2"
// list of ids.
param4?: Array<string>
// from datetime. The format is: date-time
param5?: string
// The format is: date-time
param6?: string
// Description of param 7. The format is: date-time or date
param7?: string
// description of param8. Maximum=36.
param8?: number
// description of param 9. Minimum=1.
param9?: number
// Number of page that should be returned.
person?: {
    name?: string    // name of person.
    age?: number    // age of person.
    // extra information of this person.
    extra_info?: {
        school?: string    // school of this person.
        // job of this person.
        job: {
            salary: number    // salary per month.
            title: string    // position in company.
            full_time?: boolean    // is this person full-time or not. Default value=True.
        }
    }
}
// description of param 10.
param10?: Array<{
    search?: string    // this is search param.
    category?: string
}>
// Description of param 11.
param11?: Array<string>
param12?: Array<"bungalow" | "detached" | "flat" | "land" | "park home" | "semi-detached" | "terraced">
param13?: Array<1 | 2 | 3 | 4 | 5>
}) => any;

} // namespace functions"""

        actual_output = generate_schema_from_functions(functions)
        self.assertEqual(actual_output.strip(), expected_output.strip())


if __name__ == "__main__":
    unittest.main()
