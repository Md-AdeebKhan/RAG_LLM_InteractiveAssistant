from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# Load environment variabltelles
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Load PDF document
loader = PyPDFLoader(r"D:\PDFs\question and answers.pdf")
docs = loader.load()

# Split PDF into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
docs_chunks = splitter.split_documents(docs)

# Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Switch to this when building real applications
vectorstore = Chroma.from_documents(
    documents=docs_chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# No manual save needed - it auto-persists!retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Add memory for conversation context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create QA chain (default prompt)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

# Streamlit UI
st.title("RAG Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.text_input("Enter your question:")

if question:
    answer = qa_chain({"question": question})["answer"]
    st.session_state.chat.append((question, answer))

st.write("### Chat History")
for q, a in st.session_state.chat:
    st.write(f"user: {q}")
    st.write(f"response: {a}")
