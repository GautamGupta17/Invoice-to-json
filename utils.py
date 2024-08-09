import streamlit as st
import json
from io import BytesIO
from PIL import Image
import pytesseract
from typing import List, Dict
from langchain.document_loaders.pdf import PyPDFDirectoryLoader  # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
import os
import shutil

load_dotenv()

# Define paths and constants
CHROMA_PATH = "chroma"
embedding_function = OpenAIEmbeddings()

# Define Document class
class Document:
    def __init__(self, metadata: Dict[str, str], page_content: str):
        self.metadata = metadata
        self.page_content = page_content

    def __repr__(self):
        return f"Document(metadata={self.metadata}, page_content={self.page_content[:100]}...)"

# Function to process images
def load_document_image(image_file: BytesIO) -> List[Document]:
    file_type = image_file.type.split("/")[1]
    results = []

    if file_type in ["png", "jpg"]:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        text_lines = text.splitlines()
        newline_separated_text = '\n'.join(text_lines)

        metadata = {
            'source': image_file.name,
            'page': 0
        }

        document = Document(metadata=metadata, page_content=newline_separated_text)
        results.append(document)
    
    return results

# Function to load PDF documents
def load_documents_pdf(uploaded_file):
    temp_file = "./files/temp.pdf"
    
    
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
    
    document_loader = PyPDFDirectoryLoader(r"./files")
    
    return document_loader.load()

# Function to split text into chunks
def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    return chunks

def remove_existing_chroma_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def initialize_chroma():
    remove_existing_chroma_db()  

    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Create or reinitialize the global instance
chroma_db = initialize_chroma()

def save_to_chroma(chunks: List[Document]):
    global chroma_db
    
    if chroma_db is None:
        chroma_db = initialize_chroma()
    
    chroma_db.add_documents(chunks)
    chroma_db.persist()

# Define prompt template
PROMPT_TEMPLATE = """
Extract the following details from the text and provide them in JSON format:
- Customer Details: Name, Billing Address, Shipping Address, Phone Number, Email
- Invoice Details: Invoice Number, Date, Place of Supply, Enquire ID
- Product Details: Item Name, HSN/SAC Code, Rate, Quantity, Amount Before Tax, IGST
- Totals: Total Before Tax, Total Tax Amount (IGST), Total After Tax, Round Off, Total Amount Payable
- Bank Details: Bank Name, Account Number, IFSC Code, Branch, UPI ID, Beneficiary Name
- TCS: Rate, Amount
- Additional Notes: Amount in Words, Return Policy

Text: {context}

Answer the question based on the above context: {question}
"""

query_text = "Give all the customer details, product name and total amount from the invoice"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def process_pdf(uploaded_file):
    documents = load_documents_pdf(uploaded_file)
    chunks = split_text(documents)
    save_to_chroma(chunks)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get model response
    # model = ChatOpenAI()
    model = HuggingFaceHub(
        repo_id = "mistralai/Mistral-7B-v0.1",
        model_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "repetition_penalty": 1.0,
            "return_full_text": False,
        }
    )
    response_text = model.predict(prompt)
    
    # Collect sources from metadata
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    # Format the response
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

# Process image files
def process_image(uploaded_file):
    documents = load_document_image(uploaded_file)
    chunks = split_text(documents)
    save_to_chroma(chunks)

    
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # model = ChatOpenAI()
    model = HuggingFaceHub(
        repo_id = "mistralai/Mistral-7B-v0.1",
        model_kwargs = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "repetition_penalty": 1.0,
            "return_full_text": False,
        }
    )
    
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

