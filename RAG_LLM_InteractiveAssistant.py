import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

# -------------------------------
# STREAMLIT APP TITLE
# -------------------------------
st.set_page_config(page_title="📘 RAG LLM Interactive Assistant", layout="wide")
st.title("📘 RAG LLM Interactive Assistant")
st.write("Upload a PDF and chat with its content using Llama 3.1 (Groq).")

# -------------------------------
# LOAD API KEY FROM STREAMLIT SECRETS
# -------------------------------
groq_api_key = st.secrets.get("api", {}).get("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("❌ `GROQ_API_KEY` not found in Streamlit secrets. Please configure it in `.streamlit/secrets.toml`.")
    st.stop()

# -------------------------------
# INITIALIZE LLM (Groq)
# -------------------------------
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing Groq model: {e}")
    st.stop()

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# PDF UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("⏳ Processing PDF..."):
        try:
            # 1️⃣ Save the uploaded file temporarily
            temp_path = "uploaded_temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2️⃣ Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # 3️⃣ Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs_chunks = text_splitter.split_documents(docs)

            # 4️⃣ Create embeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./huggingface_cache"
            )

            # 5️⃣ Create vector store
            vectorstore = Chroma.from_documents(
                documents=docs_chunks,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # 6️⃣ Memory for conversation context
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # 7️⃣ Create the QA chain
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=False
            )

            st.success("✅ PDF processed successfully! You can now ask questions below.")

        except Exception as e:
            st.error(f"❌ Error while processing the PDF: {e}")

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

                # Save to chat history
                st.session_state.chat_history.append((user_question, answer))

            except Exception as e:
                st.error(f"⚠️ Error generating answer: {e}")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("🧠 Chat History")
        for q, a in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
else:
    st.info("📤 Upload a PDF to begin chatting with your document.")
