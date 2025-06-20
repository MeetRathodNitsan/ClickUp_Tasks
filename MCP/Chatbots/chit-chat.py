from pinecone import Pinecone, ServerlessSpec
import streamlit as st
import requests
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

# ✅ Pinecone Configuration
PINECONE_API_KEY = "pcsk_5WWFnv_Q2sSStLJ2BKhgCJKbvMfTH5JF8bWRhqR3p2eZuM8yHqJvz16R7dScf9Sw5AoXsV"
PINECONE_ENV = "us-west-1"
INDEX_NAME = "my-vector-index"
DIMENSION = 384  # for 'all-MiniLM-L6-v2'

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Connect to index
index = pc.Index(INDEX_NAME)

# ✅ Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Streamlit UI setup
st.set_page_config(page_title="My Chat", layout="centered")
st.title("Chat with chit-chat (MCP + Pinecone)")
MEMORY_FILE = "chat_memory.json"

# ✅ Load/save chat history
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_memory(history):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(history, f)

# ✅ Build user context from local chat history
def build_context(query, memory):
    query_vec = model.encode(query)
    best_msg = ""
    best_score = 0.0
    for role, msg in memory:
        if role == "user":
            msg_vec = model.encode(msg)
            score = cosine_similarity([query_vec], [msg_vec])[0][0]
            if score > best_score:
                best_score = score
                best_msg = msg
    return best_msg if best_score > 0.6 else ""

# ✅ Retrieve from Pinecone
def retrieve_pinecone_context(query, top_k=3):
    query_vec = model.encode(query).tolist()
    result = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in result["matches"] if "text" in match["metadata"]])

# ✅ Upsert messages to Pinecone
def store_messages_to_pinecone(messages):
    to_upsert = []
    for i, (role, msg) in enumerate(messages):
        if role == "user":
            embedding = model.encode(msg).tolist()
            to_upsert.append({
                "id": f"msg-{i}",
                "values": embedding,
                "metadata": {"text": msg}
            })
    if to_upsert:
        index.upsert(vectors=to_upsert)

# ✅ Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "Meet",
        "tone": "friendly"
    }

# ✅ Display history (for devs)
if st.session_state.chat_history:
    st.subheader("💬 Chat History (Only for Developer)")
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

# ✅ Handle user input
if prompt := st.chat_input("Ask me Anything...."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append(("user", prompt))

    # ✅ Upsert to Pinecone
    store_messages_to_pinecone(st.session_state.chat_history)

    # ✅ Local and Pinecone context
    context = build_context(prompt, st.session_state.chat_history)
    pinecone_context = retrieve_pinecone_context(prompt)

    # ✅ Prepare message for LLM
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant talking to {st.session_state.user_profile['name']} in a {st.session_state.user_profile['tone']} tone."
        }
    ]

    if context:
        messages.append({"role": "system", "content": f"Relevant user context: \"{context}\""})

    if pinecone_context:
        messages.append({"role": "system", "content": f"External knowledge: \"{pinecone_context}\""})

    for role, msg in st.session_state.chat_history:
        messages.append({"role": role, "content": msg})

    messages.append({"role": "user", "content": prompt})

    # ✅ LLM API call (e.g., Ollama)
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:1b",
                "messages": messages,
                "stream": True
            },
            stream=True
        )

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    full_response += content
                    response_container.markdown(full_response)

            st.session_state.chat_history.append(("assistant", full_response))
            save_memory(st.session_state.chat_history)

    except Exception as e:
        st.error(f"Error: {e}")
