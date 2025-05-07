import os
import tempfile
import gradio as gr
import warnings
import logging
import io
import json
from tqdm import tqdm
import shutil
from datetime import datetime
import time
import re

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import components from our modules
from config import *
from agents import *
from ui import *

# Import required libraries
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as CrewAITool


# Initialize chat history
chat_history = []

# Create a string handler to capture logs for display in the UI
class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_output = io.StringIO()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        self.log_output.write(self.format(record) + '\n')

    def get_logs(self):
        return self.log_output.getvalue()

    def clear(self):
        self.log_output = io.StringIO()

# Create and add the string handler
string_handler = StringIOHandler()
logger.addHandler(string_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)



# Initialize the vector store
vector_store = None

# Initialize document metadata storage
document_metadata = {}

# Create search tool instance
search_tool = DuckDuckGoSearchRun()

# Define tools
@CrewAITool
def search_vectorstore(query: str) -> str:
    """Search the vector store for relevant documents about IBM error codes."""
    global vector_store
    if vector_store is None:
        return "Vector store is not initialized. Please upload documents first."
    
    try:
        docs = vector_store.similarity_search(query, k=3)
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown source")
            results.append(f"Document {i+1} (from {source}):\n{doc.page_content}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error searching vector store: {str(e)}"

@CrewAITool
def web_search(query: str) -> str:
    """Search the web for information about IBM error codes."""
    try:
        return search_tool.run(f"IBM error code {query}")
    except Exception as e:
        return f"Error searching the web: {str(e)}"

@CrewAITool
def format_content(content: str) -> str:
    """Format technical content about IBM error codes for better readability."""
    try:
        formatted = llm.invoke(
            f"""Format the following technical content about IBM error codes for better readability.
            Use proper markdown with headings, lists, and code blocks where appropriate:
            
            {content}"""
        )
        return formatted.content
    except Exception as e:
        return f"Error formatting content: {str(e)}\n\nOriginal content:\n{content}"

# PDF processing function
def process_pdf(file_path):
    """Extract text from PDF and split into chunks"""
    start_time = time.time()

    # Extract text from PDF
    text = ""
    logger.info(f"Starting to process PDF: {os.path.basename(file_path)}")

    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"PDF has {total_pages} pages")

        for i, page in enumerate(tqdm(pdf_reader.pages, desc="Reading PDF pages", unit="page")):
            page_text = page.extract_text()
            text += page_text
            if (i + 1) % 5 == 0 or (i + 1) == total_pages:
                progress_percentage = ((i + 1) / total_pages) * 100
                logger.info(f"Processed {i + 1}/{total_pages} pages ({progress_percentage:.1f}%)")

    logger.info(f"PDF text extraction completed. Total characters: {len(text)}")

    # Split text into chunks
    logger.info("Starting text chunking...")
    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
    chunks = text_splitter.split_text(text)
    logger.info(f"Text splitting completed. Created {len(chunks)} chunks")

    # Create documents
    logger.info("Converting chunks to Document objects...")
    documents = [Document(page_content=chunk, metadata={"source": os.path.basename(file_path)}) for chunk in chunks]

    processing_time = time.time() - start_time
    logger.info(f"PDF processing completed in {processing_time:.2f} seconds")

    return documents, total_pages

# Process uploaded files and create vector store
def process_files(files):
    global vector_store, document_metadata

    # Clear previous logs
    string_handler.clear()

    # Stats dictionary to track processing metrics
    stats = {
        "file_count": len(files),
        "total_pages": 0,
        "total_chunks": 0,
        "processing_times": {},
        "embedding_times": {},
        "total_time": 0
    }

    start_time = time.time()
    all_documents = []

    logger.info(f"Starting to process {len(files)} files")

    for i, file in enumerate(files):
        file_name = os.path.basename(file.name)
        file_start_time = time.time()
        logger.info(f"Processing file {i+1}/{len(files)}: {file_name}")

        documents, page_count = process_pdf(file.name)
        all_documents.extend(documents)

        # Update stats
        stats["total_pages"] += page_count
        file_time = time.time() - file_start_time
        stats["processing_times"][file_name] = {
            "time_seconds": round(file_time, 2),
            "pages": page_count,
            "chunks": len(documents)
        }
        stats["total_chunks"] += len(documents)

        # Add to document metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        document_metadata[file_name] = {
            "filename": file_name,
            "upload_time": timestamp,
            "page_count": page_count,
            "chunk_count": len(documents),
            "processing_time": round(file_time, 2)
        }

        logger.info(f"Completed file {i+1}/{len(files)} in {file_time:.2f} seconds")

        # Update logs and stats in UI
        yield string_handler.get_logs(), "", stats, get_indexed_documents_html()

    logger.info(f"All files processed. Total documents: {len(all_documents)}")

    # Create vector store
    logger.info("Starting to create FAISS vector store...")
    embedding_start = time.time()

    # Count total tokens for embedding
    total_tokens = sum(len(doc.page_content.split()) for doc in all_documents)
    logger.info(f"Creating embeddings for {len(all_documents)} documents (approx. {total_tokens} tokens)")

    # Process in batches to show progress
    batch_size = min(100, max(1, len(all_documents) // 10))

    all_batches = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]

    # Initialize stats for vector store creation
    stats["vector_store"] = {
        "batch_count": len(all_batches),
        "batch_size": batch_size,
        "total_documents": len(all_documents),
        "total_tokens": total_tokens,
        "batch_progress": 0
    }

    for i, batch in enumerate(tqdm(all_batches, desc="Creating vector store", unit="batch")):
        batch_start = time.time()
        if i == 0 and vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
            logger.info(f"Initial vector store created with {len(batch)} documents")
        else:
            vector_store.add_documents(batch)
            logger.info(f"Added batch {i+1}/{len(all_batches)} to vector store. Progress: {((i+1) / len(all_batches)) * 100:.1f}%")

        # Update batch stats
        batch_time = time.time() - batch_start
        stats["embedding_times"][f"batch_{i+1}"] = round(batch_time, 2)
        stats["vector_store"]["batch_progress"] = round((i + 1) / len(all_batches) * 100, 1)

        # Update logs in UI periodically
        if (i + 1) % 2 == 0 or (i + 1) == len(all_batches):
            yield string_handler.get_logs(), "", stats, get_indexed_documents_html()

    embedding_time = time.time() - embedding_start
    total_time = time.time() - start_time

    # Update final stats
    stats["embedding_total_time"] = round(embedding_time, 2)
    stats["total_time"] = round(total_time, 2)
    stats["documents_per_second"] = round(len(all_documents) / total_time, 2)

    logger.info(f"Vector store creation completed in {embedding_time:.2f} seconds")
    logger.info(f"Total processing completed in {total_time:.2f} seconds")

    # Save vector store
    save_result = save_vector_store()
    if save_result:
        logger.info("Vector store saved to disk successfully")
    else:
        logger.warning("Failed to save vector store to disk")

    completion_message = f"Processed {len(files)} files with {len(all_documents)} chunks in {total_time:.2f} seconds"
    logger.info(completion_message)

    yield string_handler.get_logs(), completion_message, stats, get_indexed_documents_html()

# CrewAI Research Agent
def research_agent(query):
    global vector_store, chat_history

    if vector_store is None:
        return "Please upload IBM Error code PDF documents first."

    # Start query stats
    query_stats = {
        "query": query,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_time": 0,
        "research_time": 0,
        "format_time": 0,
        "total_time": 0,
        "retrieved_documents": 0
    }
    query_start = time.time()

    logger.info(f"Processing query: {query}")

    try:
        # Get context from vector store
        logger.info("Retrieving relevant documents from vector store...")
        retrieval_start = time.time()
        docs = vector_store.similarity_search(query, k=5)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"Retrieved {len(docs)} documents in {retrieval_time:.2f} seconds")

        # Update query stats
        query_stats["retrieval_time"] = round(retrieval_time, 2)
        query_stats["retrieved_documents"] = len(docs)

        # Log document sources
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown source")
            logger.info(f"Document {i+1}: from {source}, starts with: {doc.page_content[:50]}...")

        # Prepare context for the agent
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create CrewAI agent
        logger.info("Creating CrewAI research agent...")
        research_start = time.time()
        
        ibm_expert = create_ibm_expert_agent()
        research_task = create_research_task(query, context, chat_history,ibm_expert)
        
        # Create and run the crew
        crew = Crew(
            agents=[ibm_expert],
            tasks=[research_task],
            verbose=True,
            process=Process.sequential
        )
        
        try:
            result = crew.kickoff()
        except Exception as e:
            logger.warning(f"Agent execution encountered an issue: {str(e)}")
            result = "The query processing encountered an issue. Please try a more specific question."
        
        research_time = time.time() - research_start
        logger.info(f"Research completed in {research_time:.2f} seconds")
        
        # Update query stats
        query_stats["research_time"] = round(research_time, 2)
        query_stats["total_time"] = round(time.time() - query_start, 2)
        
        # Log query stats
        logger.info(f"Query stats: {query_stats}")
        
        # Update chat history
        chat_history.append((query, result))
        clean_response = result.raw
        return clean_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_message = f"An error occurred while processing your query: {str(e)}. Please try again with a different question."
        chat_history.append((query, error_message))
        return error_message

def sanitize_for_chatbot(markdown_text: str) -> str:
    # Replace markdown headers (###, ##, #) with bold
    response = markdown_text.replace("### ", "**").replace("## ", "**")
    return response

# Chat function
def chat(message, history):
    if vector_store is None:
        logger.warning("Attempted to chat without uploading documents first")
        error_message = "Please upload IBM Error code PDF documents first."
        return history + [[message, error_message]]

    # Clear previous logs for new query
    string_handler.clear()

    logger.info(f"New chat message received: {message[:50]}..." if len(message) > 50 else f"New chat message received: {message}")
    response = research_agent(message)
    logger.info("Chat response generated successfully")

    # Update log display
    if 'log_output' in globals():
        logger.info(string_handler.get_logs())

    # Return updated history with new message and response
    return history + [[message, response]]

# Clear vector store
def clear_vector_store():
    global vector_store, document_metadata
    
    logger.info("Clearing vector store and metadata...")
    
    # Clear in-memory objects
    vector_store = None
    document_metadata = {}
    
    # Remove directory contents
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
        os.makedirs(VECTOR_STORE_DIR)
    
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
    
    logger.info("Vector store and metadata cleared successfully")
    
    return string_handler.get_logs(), "Vector store cleared successfully", {}, "<p>No documents indexed yet.</p>"

def refresh_log_display():
    return string_handler.get_logs()

def clear_chat_and_logs():
    string_handler.clear()
    logger.info("Chat history cleared")
    return [], string_handler.get_logs()

def main():
    logger.info("Starting IBM Error Code PDF Chat Application")
    
    # Try to load existing vector store
    load_success = load_vector_store()
    if load_success:
        logger.info(f"Successfully loaded existing vector store with {len(document_metadata)} documents")
    else:
        logger.info("No existing vector store found or failed to load")
    
    # Create UI
    demo = create_ui(
        process_files_fn=process_files,
        chat_fn=chat,
        clear_vector_store_fn=clear_vector_store,
        refresh_logs_fn=refresh_log_display,
        clear_chat_and_logs_fn=clear_chat_and_logs
    )
    
    demo.launch()
    logger.info("Application shutdown")

if __name__ == "__main__":
    main()