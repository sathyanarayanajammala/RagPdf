from crewai import Agent, Task
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from config import VECTOR_STORE_DIR, METADATA_FILE
import json
import logging
from tools import search_vectorstore, web_search, format_content  # Import tools here
from config import llm 
logger = logging.getLogger('ibm_error_code_rag')

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the vector store and metadata
vector_store = None
document_metadata = {}


def create_ibm_expert_agent():
    """Create and return the IBM Error Code Expert agent"""
    return Agent(
        role="IBM Error Code Expert",
        goal="Find and explain IBM error codes accurately and clearly",
        backstory="""You are an expert in IBM error codes with years of experience 
        troubleshooting enterprise systems. Your expertise helps developers and 
        system administrators quickly resolve issues.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_vectorstore, web_search, format_content],
        llm=llm,
        max_iter=5
    )

def create_research_task(query, context, chat_history,ibm_expert):
    """Create and return a research task for the agent"""
    return Task(
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

def load_vector_store():
    global vector_store, document_metadata
    
    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        try:
            logger.info("Found existing vector store. Loading...")
            vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
            
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

def save_vector_store():
    global vector_store, document_metadata
    
    if vector_store:
        try:
            logger.info("Saving vector store...")
            vector_store.save_local(VECTOR_STORE_DIR)
            
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