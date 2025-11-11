from langchain_groq import ChatGroq
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Use the specialized Chroma package
from langchain.memory import ConversationBufferMemory # This one is still in core 'langchain'
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
groq_api_key = st.secrets["api"]["GROQ_API_KEY"]

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
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
                loader = PyPDFLoader("temp.pdf")
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
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=False
            )
            st.success("PDF processed successfully. you can now ask questions...")

        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")

# UI for asking questions
if st.session_state.qa_chain is not None:
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain({"question": question})
            answer = result["answer"]

            # Store in chat history
            st.session_state.chat.append((question, answer))

    # Display chat history
    if st.session_state.chat:
        st.write("### Chat History")
        for q, a in st.session_state.chat:
            st.write(f"**You:** {q}")
            st.write(f"**Assistant:** {a}")
else:
    st.info("Upload a PDF to begin.")


