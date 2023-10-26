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
                                                "positions": {"type": "array", "items": {"type": "string"}},
                                                "full_time": {
                                                    "type": "boolean",
                                                    "description": "is this person full-time or not",
                                                    "default": True,
                                                },
                                                "ids": {
                                                    "type": "array",
                                                    "description": "ids of this job",
                                                    "items": {"type": "number", "description": "ids for this job hehe"},
                                                },
                                                "params": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "a1": {"type": "string", "description": "a1"},
                                                                "a2": {"type": "string", "description": "a2"},
                                                                "a3": {"type": "string", "description": "a3"},
                                                            },
                                                        },
                                                    },
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
                        "param140": {"type": ["number", "null"]},
                        "param14": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string", "description": "array of array"}},
                        },
                        "param15": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "description": "array of array",
                                    "properties": {
                                        "att1": {"type": "string", "description": "desc 1"},
                                        "att2": {"type": "number", "description": "desc 2"},
                                    },
                                },
                            },
                        },
                    },
                    "required": ["param1", "param2", "param15"],
                },
            }
        ]

        expected_output = """// Supported function definitions that should be called when necessary.
namespace functions {

// This is a test function
type test_function = (_: {
param1: string,
// description of param 2.
param2: string,
// description of param 3. Default="option1".
param3?: "option1" | "option2",
// list of ids.
param4?: string[],
// from datetime. Format=date-time
param5?: string,
// Format=date-time
param6?: string,
// Description of param 7. Format=date-time or date
param7?: string,
// description of param8. Maximum=36
param8?: number,
// description of param 9. Minimum=1
param9?: number,
// Number of page that should be returned.
person?: {
    // name of person.
    name?: string,
    // age of person.
    age?: number,
    // extra information of this person.
    extra_info?: {
        // school of this person.
        school?: string,
        // job of this person.
        job: {
            // salary per month.
            salary: number,
            // position in company.
            title: string,
            positions?: string[],
            // is this person full-time or not. Default=True.
            full_time?: boolean,
            // ids of this job.
            ids?: number[],
            params?: {
                    // a1.
                    a1?: string,
                    // a2.
                    a2?: string,
                    // a3.
                    a3?: string,
                }[][],
        },
    },
},
// description of param 10.
param10?: {
    // this is search param.
    search?: string,
    category?: string,
}[],
// Description of param 11.
param11?: string[],
param12?: ("bungalow" | "detached" | "flat" | "land" | "park home" | "semi-detached" | "terraced")[],
param13?: (1 | 2 | 3 | 4 | 5)[],
param140?: number | null,
param14?: string[][],
param15: {
        // desc 1.
        att1?: string,
        // desc 2.
        att2?: number,
    }[][],
}) => any;

} // namespace functions"""

        actual_output = generate_schema_from_functions(functions)
        self.assertEqual(actual_output.strip(), expected_output.strip())


if __name__ == "__main__":
    unittest.main()
