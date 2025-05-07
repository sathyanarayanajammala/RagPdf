from crewai import Agent, Task
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from config import VECTOR_STORE_DIR, METADATA_FILE
import json
import logging
from tools import search_vectorstore, web_search, format_content  # Import tools here
from config import llm, PROMPTS   
logger = logging.getLogger('ibm_error_code_rag')

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the vector store and metadata
vector_store = None
document_metadata = {}


def create_ibm_expert_agent():
    """Create and return the IBM Error Code Expert agent"""
    return Agent(
        role=PROMPTS["agent"]["ibm_expert"]["role"],
        goal=PROMPTS["agent"]["ibm_expert"]["goal"],
        backstory=PROMPTS["agent"]["ibm_expert"]["backstory"],
        verbose=True,
        allow_delegation=False,
        tools=[search_vectorstore, format_content],
        llm=llm,
        max_iter=5
    )

def create_research_task(query, context, chat_history, agent):
    # Ensure all required parameters are passed
    if not all([query, agent]):
        raise ValueError("Missing required parameters for task creation")
    
    description = PROMPTS["task"]["research"].format(
        query=query,
        context=context or "No relevant documents found",
        chat_history=chat_history or "No prior conversation"
    )
    
    return Task(
        description=description,
        agent=agent,
        expected_output=PROMPTS["task"]["expected_output"]
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
            
            # Ensure the directory exists
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            
            # Save the vector store
            vector_store.save_local(VECTOR_STORE_DIR)
            
            # Ensure the directory for metadata file exists
            metadata_dir = os.path.dirname(METADATA_FILE)
            if metadata_dir:
                os.makedirs(metadata_dir, exist_ok=True)
            
            # Save metadata
            with open(METADATA_FILE, 'w') as f:
                json.dump(document_metadata, f)
                
            logger.info(f"Vector store saved to {VECTOR_STORE_DIR}")
            logger.info(f"Metadata saved to {METADATA_FILE}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    else:
        logger.warning("No vector store to save")
        return False