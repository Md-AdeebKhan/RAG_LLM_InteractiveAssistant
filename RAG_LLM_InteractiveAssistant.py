from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# Load environment variables and setup Groq LLM
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Load and process the PDF document
loader = PyPDFLoader("D:\\health care pdf\\diabetes-training-manual.pdf")
docs = loader.load()

# Split document into chunks for processing
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
docs_chunks = splitter.split_documents(docs)

# Create embeddings and vector store for similarity search
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs_chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Simple prompt template for the LLM
prompt = PromptTemplate.from_template("""
Context: {context}
Question: {question}
Answer:
""")

# Memory to remember conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Main chain that combines everything - LLM, retrieval, and memory
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    output_key="answer",
    return_source_documents=False
)

# Streamlit web interface
st.title("RAG LLM Interactive Assistant")

# Keep track of chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input field
user_input = st.text_input("You:")

# When user asks a question
if user_input:
    # Get answer from our AI chain
    response = qa_chain({"question": user_input})
    
    # Save the conversation
    st.session_state.history.append((user_input, response["answer"]))

# Show the conversation history
for user_msg, bot_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")