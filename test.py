import requests

payload = {
    "jsonrpc": "2.0",
    "method": "callTool",
    "params": {
        "tool": "list_files",
        "arguments": {}
    },
    "id": 1
}

response = requests.post("http://127.0.0.1:8000/", json=payload)
print(response.json())