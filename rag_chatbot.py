import os
import getpass
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import datetime
import warnings
import logging

# Suppress warnings/logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('cassandra.protocol').setLevel(logging.ERROR)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=4000
)

# Local Vector store using Chroma
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Better text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

# Process a single text file
def process_txt(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        if chunk.strip():
            content = chunk.strip()
            metadata = {"source": os.path.basename(file_path), "type": "txt"}
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

# Process all .txt files from a directory
def process_directory(directory_path):
    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            docs = process_txt(file_path)
            all_docs.extend(docs)
    return all_docs

# Process entire directory
directory_path = '/home/sracha/Sattvastha/RAG_Nikita/documents/'
docs = process_directory(directory_path)

# Add all documents to vector store
vector_store.add_documents(docs)
print(f"Inserted {len(docs)} documents from directory.")
vector_store.persist()

# Wrap in vector store index
vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

# Chat memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Custom prompt to prevent "according to the text" type answers
custom_prompt_template = """
You are a mental health assistant having a conversation with a user.
Answer the user's question naturally based on the given context.
Do NOT mention that the information came from any document or context.
Simply answer as if you know the information.

Context:
{context}

User Question:
{question}

Helpful Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template,
)

# RAG chain setup with custom prompt
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Message formatting
def format_message(role, content):
    color = "#4a76a8" if role == "Bot" else "#6c757d"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: {color}; color: white;">
        <strong>{role}</strong> - {timestamp}<br>
        {content}
    </div>
    """

# Chat loop
def chat_with_rag():
    print("Hi, how are you feeling today? You can type 'exit' or 'thankyou' to end the conversation.")
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['exit', 'thankyou']:
            print("Bot: Goodbye! Have a great day!")
            break

        if not user_input.strip():
            print("Bot: Please enter a valid question.")
            continue

        try:
            result = qa_chain.invoke({"question": user_input})
            print(f"Bot: {result['answer']}")
        except Exception as e:
            print(f"Bot: Oops! Something went wrong. Error: {str(e)}")

if __name__ == "__main__":
    chat_with_rag()
