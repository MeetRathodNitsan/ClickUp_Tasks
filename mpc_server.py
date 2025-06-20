import os
import json
import requests


TOOLS = {
    "list_files": {
        "description": "List files in the current folder",
        "parameters": {}
    },
    "read_file": {
        "description": "Read a file",
        "parameters": {"path": "Path of the file"}
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": {"path": "Path to the file", "content": "Content to write"}
    },
    "modify_file": {
        "description": "Modify a file",
        "parameters": {"path": "Path to the file", "content": "Content to write"}
    },
    "ask_llm": {
        "description": "Ask a prompt to your local Ollama LLM",
        "parameters": {"prompt": "Your prompt text"}
    }
}


def list_files():
    return os.listdir()


def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"❌ Error: {e}"


def write_file(path, content):
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"✅ File '{path}' written successfully."
    except Exception as e:
        return f"❌ Error: {e}"

def modify_file(path, content):
    try:
        with open(path, "w+") as f:
            f.write(content)
        return F"✅ File '{path}' over-written successfully."
    except Exception as e:
        return f"❌ Error: {e}"


def ask_llm(prompt):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(url, json=payload)
        return res.json().get("response", "⚠️ No response from model.")
    except Exception as e:
        return f"❌ Ollama error: {e}"


def run_terminal_mcp():
    print("🧠 MCP Terminal Agent Started!\nAvailable tools:")
    for tool_name, details in TOOLS.items():
        print(f"• {tool_name}: {details['description']}")

    while True:
        cmd = input("\n👉 Enter tool name (or 'exit' to quit): ").strip()

        if cmd == "exit":
            print("👋 Exiting MCP Agent.")
            break

        if cmd not in TOOLS:
            print("❌ Unknown tool. Try again.")
            continue

        args = {}
        for param in TOOLS[cmd]["parameters"]:
            args[param] = input(f"Enter value for '{param}': ")

        if cmd == "list_files":
            result = list_files()
        elif cmd == "read_file":
            result = read_file(args["path"])
        elif cmd == "write_file":
            result = write_file(args["path"], args["content"])
        elif cmd == "modify_file":
            result = modify_file(args["path"], args["content"])
        elif cmd == "ask_llm":
            result = ask_llm(args["prompt"])
        else:
            result = "❌ Not implemented."

        print("\n📤 Result:\n", result)


if __name__ == "__main__":
    run_terminal_mcp()
