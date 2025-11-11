import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
import os

# -------------------------------
# STREAMLIT APP CONFIG
# -------------------------------
st.set_page_config(page_title="📘 RAG LLM Interactive Assistant", layout="wide")
st.title("📘 RAG LLM Interactive Assistant")
st.write("Upload a PDF and chat with it using Llama 3.1 (Groq).")

# -------------------------------
# LOAD API KEY
# -------------------------------
groq_api_key = st.secrets.get("api", {}).get("GROQ_API_KEY", None)
if not groq_api_key:
    st.error("❌ `GROQ_API_KEY` not found in Streamlit secrets. Please configure it.")
    st.stop()

# -------------------------------
# INITIALIZE LLM
# -------------------------------
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing Groq model: {e}")
    st.stop()

# -------------------------------
# SESSION STATE
# -------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("⏳ Processing PDF..."):
        try:
            # Save uploaded file
            temp_path = "uploaded_temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs_chunks = text_splitter.split_documents(docs)

            # Embeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./huggingface_cache"
            )

            # Vector store
            vectorstore = Chroma.from_documents(
                documents=docs_chunks,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                chat_memory=ChatMessageHistory()
            )

            # QA Chain
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=False
            )

            st.success("✅ PDF processed successfully! You can now ask questions below.")

        except Exception as e:
            st.error(f"❌ Error while processing PDF: {e}")

# -------------------------------
# CHAT INTERFACE
# -------------------------------
if st.session_state.qa_chain:
    user_question = st.text_input("💬 Ask a question about your PDF:")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.qa_chain({"question": user_question})
                answer = response["answer"]
                st.session_state.chat_history.append((user_question, answer))
            except Exception as e:
                st.error(f"⚠️ Error generating answer: {e}")

    if st.session_state.chat_history:
        st.subheader("🧠 Chat History")
        for q, a in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
else:
    st.info("📤 Upload a PDF to begin chatting with your document.")
