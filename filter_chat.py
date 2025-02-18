import json 
import typer


def main(data_path: str, output_path: str):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    result = []
    for item in data:
        messages = item.get("messages", []) or []
        tools = item.get("tools", []) or []
        if len(tools) == 0:
            result.append(item)

    print(f"Filtered {len(result)}/{len(data)} items")
    with open(output_path, "w") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    typer.run(main)
