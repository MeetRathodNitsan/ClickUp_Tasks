import os
import re
import json
import requests
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS

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
    },
    "search_and_download_pdf": {
    "description": "Search and auto-download first PDF for query",
    "parameters": {"query": "Search keywords for PDF"}
    },
    "summarize_pdf": {
        "description": "Summarize content from a PDF",
        "parameters": {"path": "Path to the PDF"}
    },
    "detect_intent": {
        "description": "Determine the appropriate tool for a free-form task",
        "parameters": {"prompt": "User input"}
    }
}

# === Tool Functions ===
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
        return f"✅ File '{path}' modified successfully."
    except Exception as e:
        return f"❌ Error: {e}"

def ask_llm_chat():
    print("💬 Entering LLM chat mode (type 'exit' to quit)...\n")
    history = []

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting chat.")
            break

        # Append user input to history
        history.append(f"User: {user_input}")

        # Build prompt with full history
        prompt = "\n".join(history) + "\nAI:"

        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            }
            res = requests.post(url, json=payload)
            reply = res.json().get("response", "⚠️ No response from model.")

            print("🤖 AI:", reply.strip())
            history.append(f"AI: {reply.strip()}")
        except Exception as e:
            print(f"❌ Ollama error: {e}")

def search_and_download_pdf(query):
    with DDGS() as ddgs:
        results = ddgs.text(query + " filetype:pdf", max_results=10)

    pdf_urls = [r['href'] for r in results if r['href'].lower().endswith('.pdf')]
    if not pdf_urls:
        return "❌ No PDF links found."

    for i, url in enumerate(pdf_urls, start=1):
        try:
            print(f"🔍 Trying URL {i}: {url}")
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()

            filename = re.sub(r'\W+', '_', query)[:50] + f"_{i}.pdf"
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return f"✅ PDF downloaded: {filename} (from {url})"
        except Exception as e:
            print(f"⚠️ Skipping broken link: {url} — {e}")

    return "❌ All found links failed to download."

def summarize_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages[:3]:
            text += page.extract_text()
        return ask_llm("Summarize this:\n" + text[:2000])
    except Exception as e:
        return f"❌ PDF error: {e}"

def detect_intent(prompt):
    try:
        tool = ask_llm(f"Which tool should I use for this request: '{prompt}'? Reply ONLY with the tool name.")
        return tool.strip()
    except Exception as e:
        return f"❌ LLM error: {e}"

# === Agent Runner ===
def run_mcp():
    print("\n🤖 MCP Agent (Multi-Tool AI) is Running!")
    print("Type your request in natural language or use a specific tool name. Type 'exit' to quit.")

    while True:
        cmd = input("\n👉 Your command: ").strip()
        if cmd.lower() == "exit":
            print("👋 Goodbye!")
            break

        tool = cmd if cmd in TOOLS else detect_intent(cmd)

        if tool not in TOOLS:
            print("❌ Unknown tool or intent.")
            continue

        args = {}
        for param in TOOLS[tool]["parameters"]:
            args[param] = input(f"🔸 Enter value for '{param}': ")

        if tool == "list_files":
            result = list_files()
        elif tool == "read_file":
            result = read_file(args["path"])
        elif tool == "write_file":
            result = write_file(args["path"], args["content"])
        elif tool == "modify_file":
            result = modify_file(args["path"], args["content"])
        elif tool == "ask_llm":
            ask_llm_chat()
            continue
        elif tool == "search_and_download_pdf":
            result = search_and_download_pdf(args["query"])
        elif tool == "summarize_pdf":
            result = summarize_pdf(args["path"])
        else:
            result = "❌ Tool not implemented."

        print("\n📤 Result:\n", result)

if __name__ == "__main__":
    run_mcp()
