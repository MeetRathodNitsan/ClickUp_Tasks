from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import mysql.connector
import json


conn = mysql.connector.connect(user='root', password='', host='localhost', database='test')
cursor = conn.cursor()


model = SentenceTransformer('all-MiniLM-L6-v2')


texts_to_insert = [
    "AI is transforming our future.",
    "Cats are cute and fluffy animals.",
    "I am a computer Engineer finally."
]

for text in texts_to_insert:
   
    cursor.execute("SELECT COUNT(*) FROM documents WHERE content = %s", (text,))
    if cursor.fetchone()[0] == 0:
        embedding = model.encode(text).tolist()
        embedding_json = json.dumps(embedding)
        cursor.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (text, embedding_json))

conn.commit()


query_text = input("Type your text or sentence for finding Similarity Score! :")
query_vector = model.encode(query_text).reshape(1, -1)

cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT content, embedding FROM documents")
rows = cursor.fetchall()

results = []
for row in rows:
    stored_vector = np.array(json.loads(row['embedding'])).reshape(1, -1)
    similarity = cosine_similarity(query_vector, stored_vector)[0][0]
    results.append((row['content'], similarity))


results.sort(key=lambda x: x[1], reverse=True)

print("\n🔍 Similarity Results:")
for content, score in results:
    print(f"{content} → Similarity: {score:.4f}")


cursor.close()
conn.close()
