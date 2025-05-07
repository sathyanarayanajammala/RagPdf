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

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from dotenv import load_dotenv
import time
import re
# Import CrewAI components
from crewai import Agent, Task, Crew, Process,LLM
from crewai.tools import tool as CrewAITool  # Use the @tool decorator instead
from langchain_community.tools import DuckDuckGoSearchRun

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_MODEL_FULL_NAME = f"gemini/{LLM_MODEL_NAME}" 
# Create logger
logger = logging.getLogger('ibm_error_code_rag')

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

# Load environment variables from .env file
load_dotenv()

# Set Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Configure storage directories
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
VECTOR_STORE_DIR = os.path.join(STORAGE_DIR, "vector_store")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.json")

# Create storage directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the vector store
vector_store = None

# Initialize document metadata storage
document_metadata = {}

# Initialize chat history
chat_history = []

# Create LLM
#llm = ChatGoogleGenerativeAI(model=LLM_MODEL_FULL_NAME, temperature=0.2)
llm = LLM(
    model=LLM_MODEL_FULL_NAME,
    temperature=0.7,
    api_key=GOOGLE_API_KEY,
    # Additional recommended parameters
    max_tokens=1024,        # Control response length
    top_p=0.95,            # Nucleus sampling parameter
    top_k=40,              # Limit vocabulary diversity
    presence_penalty=0.2,   # Reduce repetition
    frequency_penalty=0.3,  # Encourage diverse language
    context_window=8192,    # Maximum context length
    streaming=True         # Enable streaming responses
)
# Create search tool instance
search_tool = DuckDuckGoSearchRun()

# Load existing vector store if available
def load_vector_store():
    global vector_store, document_metadata
    
    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        try:
            logger.info("Found existing vector store. Loading...")
            # Added allow_dangerous_deserialization=True to fix the loading error
            vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
            
            # Load document metadata if available
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'r') as f:
                    document_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(document_metadata)} documents")
            
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    else:
        logger.info("No existing vector store found")
        return False

# Save vector store and metadata
def save_vector_store():
    global vector_store, document_metadata
    
    if vector_store:
        try:
            logger.info("Saving vector store...")
            vector_store.save_local(VECTOR_STORE_DIR)
            
            # Save document metadata
            with open(METADATA_FILE, 'w') as f:
                json.dump(document_metadata, f)
                
            logger.info("Vector store and metadata saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    else:
        logger.warning("No vector store to save")
        return False

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
            if (i + 1) % 5 == 0 or (i + 1) == total_pages:  # Log every 5 pages or at the end
                progress_percentage = ((i + 1) / total_pages) * 100
                logger.info(f"Processed {i + 1}/{total_pages} pages ({progress_percentage:.1f}%)")

    logger.info(f"PDF text extraction completed. Total characters: {len(text)}")

    # Split text into chunks
    logger.info("Starting text chunking...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
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
    batch_size = min(100, max(1, len(all_documents) // 10))  # Create at least 10 batches if possible

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

# Define a vectorstore search tool for CrewAI
# Define tools using the @tool decorator
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
# CrewAI Research Agent
# CrewAI Research Agent - Fixed version
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
        docs = vector_store.similarity_search(query, k=3)
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
        
        ibm_expert = Agent(
            role="IBM Error Code Expert",
            goal="Find and explain IBM error codes accurately and clearly",
            backstory="""You are an expert in IBM error codes with years of experience 
            troubleshooting enterprise systems. Your expertise helps developers and 
            system administrators quickly resolve issues.""",
            verbose=True,
            allow_delegation=False,
            tools=[search_vectorstore, web_search, format_content],  # Use the decorated functions directly
            llm=llm,
            max_iter=5
        )
        
        # Create the research task
        research_task = Task(
            description=f"""Research and provide a detailed explanation for the following IBM error code query:
            
            QUERY: {query}
            
            CONTEXT FROM DOCUMENTS:
            {context}
            
            PREVIOUS CONVERSATION:
            {chat_history}
            
            Provide a clear, detailed explanation of the error, its causes, and step-by-step troubleshooting 
            instructions. Format your response using proper markdown with headings, lists, and code blocks 
            where appropriate. Include examples where relevant.""",
            agent=ibm_expert,
            expected_output="A comprehensive, well-formatted explanation of the IBM error code with troubleshooting steps."
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[ibm_expert],
            tasks=[research_task],
            verbose=True,
            process=Process.sequential
        )
        
        # Execute the crew
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
        logger.info(f"Finale Result Before : {result}")
        clean_response = sanitize_for_chatbot(result.raw)
        logger.info(f"clean_response : {clean_response}")
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
        
        # Return the message pair in standard format
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

# Get HTML for indexed documents
def get_indexed_documents_html():
    if not document_metadata:
        return "<p>No documents indexed yet.</p>"
    
    html = "<h3>Indexed Documents</h3>"
    html += "<table style='width:100%; border-collapse: collapse;'>"
    html += "<tr style='background-color: #f2f2f2;'>"
    html += "<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Filename</th>"
    html += "<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Upload Time</th>"
    html += "<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Pages</th>"
    html += "<th style='padding: 8px; text-align: left; border: 1px solid #ddd;'>Chunks</th>"
    html += "</tr>"
    
    for key, doc in document_metadata.items():
        html += f"<tr>"
        html += f"<td style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{doc['filename']}</td>"
        html += f"<td style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{doc['upload_time']}</td>"
        html += f"<td style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{doc['page_count']}</td>"
        html += f"<td style='padding: 8px; text-align: left; border: 1px solid #ddd;'>{doc['chunk_count']}</td>"
        html += f"</tr>"
    
    html += "</table>"
    return html

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

# Define Gradio UI
with gr.Blocks(theme="gradio/default") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("** Upload IBM Error Code PDFs")
            file_input = gr.Files(label="Upload PDF Files", file_types=[".pdf"], file_count="multiple")
            upload_button = gr.Button("Process Files")
            clear_store_button = gr.Button("Clear Vector Store", variant="secondary")
            status_text = gr.Textbox(label="Processing Status", interactive=False)

            # Add a log display area and refresh button
            with gr.Row():
                log_output = gr.Textbox(label="Processing Logs", interactive=False, lines=10, autoscroll=True)
                refresh_logs = gr.Button("Refresh Logs")

            # Add a summary statistics area
            stats_output = gr.JSON(label="Processing Statistics", visible=True)
            
            # Add indexed documents display
            indexed_docs_html = gr.HTML(label="Indexed Documents")

        with gr.Column(scale=2):
            gr.Markdown("** Chat with IBM Error Code Documents")
            # Explicitly set type='tuples' to use older format
            chatbot = gr.Chatbot(height=400, type='tuples')
            message_input = gr.Textbox(label="Ask a question about IBM error codes", placeholder="Type your question here...")
            clear_button = gr.Button("Clear Chat")

    # Set up event handlers
    upload_button.click(process_files, inputs=[file_input], outputs=[log_output, status_text, stats_output, indexed_docs_html])
    clear_store_button.click(clear_vector_store, outputs=[log_output, status_text, stats_output, indexed_docs_html])

    # Function to refresh logs
    def refresh_log_display():
        return string_handler.get_logs()

    refresh_logs.click(refresh_log_display, outputs=log_output)

    # Simple submit function using the correct format
    message_input.submit(
        fn=chat,
        inputs=[message_input, chatbot],
        outputs=[chatbot],
    ).then(
        fn=lambda: "",  # Clear the message box after sending
        outputs=[message_input]
    )

    # Update clear chat function
    def clear_chat_and_logs():
        string_handler.clear()
        logger.info("Chat history cleared")
        return [], string_handler.get_logs()

    clear_button.click(clear_chat_and_logs, outputs=[chatbot, log_output])

# Launch the app
if __name__ == "__main__":
    logger.info("Starting IBM Error Code PDF Chat Application")
    
    # Try to load existing vector store
    load_success = load_vector_store()
    if load_success:
        logger.info(f"Successfully loaded existing vector store with {len(document_metadata)} documents")
    else:
        logger.info("No existing vector store found or failed to load")
    
    demo.launch()
    logger.info("Application shutdown")