from langchain_groq import ChatGroq
import os
# Updated Imports based on LangChain Modularization
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.memory import ConversationBufferMemory # FIX: Memory moved to langchain-core
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# Load environment variable
# Assuming st.secrets["api"]["GROQ_API_KEY"] is correctly configured in Streamlit secrets
groq_api_key = st.secrets.get("api", {}).get("GROQ_API_KEY")

# Check if the API key is available before initializing the LLM
if not groq_api_key:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please configure it.")
    st.stop()
    
# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

st.title("RAG Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if uploaded_file is not None and st.session_state.qa_chain is None:
    with st.spinner("Processing PDF..."):
        try:
            # 1. Save the uploaded file temporarily
            # Using a context manager for safe file handling
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Load the document
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            # 3. Split PDF into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
            docs_chunks = splitter.split_documents(docs)

            # 4. Create embeddings and vector store
            # Ensure 'sentence-transformers' package is installed
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                # Set a path to avoid potential permission errors in Streamlit Cloud
                cache_folder="./huggingface_cache" 
            )
            
            # Ensure 'chromadb' package is installed
            vectorstore = Chroma.from_documents(
                documents=docs_chunks,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # 5. Add memory for conversation context
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # 6. Create QA chain
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=False
            )
            st.success("PDF processed successfully. You can now ask questions...")

        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")

# UI for asking questions
if st.session_state.qa_chain is not None:
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Thinking..."):
            try:
                # Use the QA chain
                result = st.session_state.qa_chain({"question": question})
                answer = result["answer"]

                # Store in chat history
                st.session_state.chat.append((question, answer))
            
            except Exception as e:
                st.error(f"Error during retrieval/generation: {e}")

    # Display chat history
    if st.session_state.chat:
        st.write("### Chat History")
        # Display history in reverse order (newest on top)
        for q, a in st.session_state.chat[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
else:
    st.info("Upload a PDF to begin.")
