from tavily import TavilyClient
import os
import json

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the web for, the query should be simple, manageable, and concise",
                }
            },
        },
    },
}


def search_tool(arguments: dict):
    print(f"search: {json.dumps(arguments, ensure_ascii=False, indent=4)}")
    query = arguments["query"]
    response = tavily_client.search(query=query)
    return json.dumps(response, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print(search_tool({"query": "weather in Singapore"}))
