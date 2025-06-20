import os
import requests
from bs4 import BeautifulSoup

# Tools registry
TOOLS = {
    "list_files": {"description": "List local directory files", "parameters": {}},
    "read_web_file": {"description": "Read file or content from a URL", "parameters": {"url": "URL to fetch"}},
    "download_web_file": {"description": "Download a file from a URL", "parameters": {"url": "File URL", "filename": "Save as filename"}},
    "write_file": {"description": "Write content to a local file", "parameters": {"path": "File path", "content": "Text content"}},
    "ask_llm": {"description": "Ask a prompt to Ollama LLM", "parameters": {"prompt": "Your prompt"}}
}

def list_files():
    return os.listdir()

def read_web_file(url):
    try:
        res = requests.get(url)
        if "text" in res.headers["Content-Type"]:
            return res.text[:2000]  # Return preview
        return f"✅ Content fetched (non-text), length: {len(res.content)} bytes"
    except Exception as e:
        return f"❌ Error reading URL: {e}"

def download_web_file(url, filename):
    try:
        res = requests.get(url)
        with open(filename, "wb") as f:
            f.write(res.content)
        return f"✅ File downloaded as '{filename}'"
    except Exception as e:
        return f"❌ Download error: {e}"

def write_file(path, content):
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"✅ File '{path}' saved."
    except Exception as e:
        return f"❌ Write error: {e}"

def ask_llm(prompt):
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        })
        return response.json().get("response", "⚠️ No response")
    except Exception as e:
        return f"❌ Ollama error: {e}"

def run_mcp_web_terminal():
    print("🌐 MCP Web Agent Started!")
    for tool, meta in TOOLS.items():
        print(f"• {tool}: {meta['description']}")

    while True:
        cmd = input("\n🔧 Tool name (or 'exit'): ").strip()
        if cmd == "exit": break
        if cmd not in TOOLS:
            print("❌ Invalid tool.")
            continue

        args = {}
        for param in TOOLS[cmd]["parameters"]:
            args[param] = input(f"→ {param}: ")

        if cmd == "list_files":
            result = list_files()
        elif cmd == "read_web_file":
            result = read_web_file(args["url"])
        elif cmd == "download_web_file":
            result = download_web_file(args["url"], args["filename"])
        elif cmd == "write_file":
            result = write_file(args["path"], args["content"])
        elif cmd == "ask_llm":
            print("🧠 You are now chatting with the local LLM (Ollama). Type 'exit' to return.")
            while True:
                user_prompt = input("🗨️ You: ")
                if user_prompt.lower() in ["exit", "quit"]:
                    break
                response = ask_llm(user_prompt)
                print("🤖 LLM:", response)

        else:
            result = "❌ Tool not implemented."

        print("\n📤 Result:\n", result)

if __name__ == "__main__":
    run_mcp_web_terminal()
