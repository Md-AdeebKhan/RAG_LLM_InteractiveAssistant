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

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

loader = PyPDFLoader("D:\\health care pdf\\diabetes-training-manual.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
docs_chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs_chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template("""
Context: {context}
Question: {question}
Answer:
""")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    output_key="answer",
    return_source_documents=False
)

import streamlit as st

# Title of the app
st.title("RAG_LLM_InteractiveAssistant")

# Initialize chat history if not already there
if "history" not in st.session_state:
    st.session_state.history = []

# Take user input
user_input = st.text_input("You:")

# If user typed something
if user_input:
    # Get bot response from your chain
    response = qa_chain({"question": user_input})
    
    # Save the conversation in the session state
    st.session_state.history.append((user_input, response["answer"]))

# Display the conversation
for user_msg, bot_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")


