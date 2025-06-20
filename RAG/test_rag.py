import requests
from pyexpat import model
from urllib3 import response
import mysql.connector
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

conn = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "test"
)

cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT id, content, embedding from documents")
rows = cursor.fetchall()

documents = []

for row in rows:
    documents.append({
        "id": row["id"],
        "content": row["content"],
        "embedding": np.array(json.loads(row["embedding"]))
    })

model = SentenceTransformer("all-MiniLM-L6-v2")
query = input("Enter your sentence here :")
query_vector = model.encode(query)

best_match = None
best_score = -1

for doc in documents:
    score = cosine_similarity([query_vector], [doc["embedding"]])[0][0]
    if score > best_score:
        best_score = score
        best_match = doc

prompt = f"""
You are an intelligent assistant. Use the context provided below to answer the question.
If the context does not contain the answer, say "I don't know."

Context:
{best_match['content']}

Question: {query}

Answer:
"""
print("=== PROMPT SENT TO MODEL ===")
print(prompt)

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.2:1b", "prompt": prompt}
)
for line in response.text.strip().split("\n"):
    if line:
        data=json.loads(line)
        print(data.get("response", ""),end="",flush=True)
