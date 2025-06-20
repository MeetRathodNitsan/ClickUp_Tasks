import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA
import tempfile



custom_prompt = PromptTemplate.from_template(
    "You are a helpful assistant who gives straight, short and sweet answers.\n\n"
    "{history}\n"
    "User: {input}\n"
    "Bot:"
)


llm = OllamaLLM(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    temperature=0.7,
    max_tokens=512,
    stream=False,
)


memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    human_prefix="User",
    ai_prefix="Bot"
)


chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt,
)

st.set_page_config(page_title="LangBot", page_icon="🤖")
st.title("🤖 Welcome to LangBot")


with st.sidebar:
    if st.button("🔄 Reset Chat"):
        st.session_state.clear()
        st.rerun()


uploaded_file = st.file_uploader("📄 Upload a file to chat with", type=["pdf", "txt", "docx"])


if "messages" not in st.session_state:
    st.session_state.messages = []


if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name


    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model="llama3.2:1b", base_url="http://localhost:11434")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()


    pdf_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    )

    st.session_state.qa_chain = pdf_chain
    st.success("✅ File uploaded and processed!")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:

        if "qa_chain" in st.session_state:
            answer = st.session_state.qa_chain.run(prompt)
        else:
            answer = chatbot.predict(input=prompt)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"❌ Error: {e}")
