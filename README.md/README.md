# RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) based chatbot built using LangChain, Groq LLM, HuggingFace embeddings, and Streamlit.  
It allows users to upload a PDF document and ask natural language questions based on the content of that document.

---

## Overview

The RAG Assistant demonstrates how to build a question-answering system that can read and understand documents.  
It follows these steps:
1. Load and process the uploaded PDF file.
2. Split the text into smaller chunks.
3. Convert those chunks into embeddings for semantic search.
4. Retrieve the most relevant parts of the text for a given question.
5. Generate an answer using the Groq LLM (Llama 3.1 model).

---

## Architecture

PDF → Loader → Text Splitter → Embeddings → Vector Store (Chroma)
↓
Retriever
↓
User Query → ConversationalRetrievalChain → Groq LLM → Response


---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Frameworks | LangChain, Streamlit |
| LLM Provider | Groq (Llama 3.1 - 8B Instant) |
| Embedding Model | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | Chroma |
| Memory | ConversationBufferMemory |
| Environment Management | python-dotenv |

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For Mac/Linux

3. Install Dependencies
pip install -r requirements.txt

4. Set Up Environment Variables
GROQ_API_KEY=your_groq_api_key_here

5. Run the Application
streamlit run app.py

Features

Upload any PDF file.
Ask context-based questions in natural language.
Maintains conversational memory for multi-turn chats.
Uses embeddings for semantic information retrieval.
Fast inference powered by Groq API.
Simple and clean Streamlit user interface.

Example Usage

Start the Streamlit app using streamlit run app.py.

Upload a PDF document.

Ask questions such as:

"What is this document about?"
"Summarize the key points in section 2."
"Who is mentioned as the author?"

The model will analyze the content of the uploaded document and provide relevant answers.

Author

Mohammed Adeeb Ahmed Khan

Email: 22.607mdadeeb@gmail.com

LinkedIn: linkedin.com/in/md-adeeb-ab849a30a

GitHub: github.com/Md-AdeebKhan


Acknowledgements

LangChain
Groq Cloud
Hugging Face
Streamlit