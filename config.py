import os
from dotenv import load_dotenv
from datetime import datetime
import logging
from crewai import LLM
# Load environment variables from .env file
load_dotenv()


# Create logger
logger = logging.getLogger('ibm_error_code_rag')
# Set Google API key


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Model Configuration
LLM_MODEL_NAME = "gemini-2.0-flash"
LLM_MODEL_FULL_NAME = f"gemini/{LLM_MODEL_NAME}"

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

# Text Splitting Configuration
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "length_function": len
}

# File Storage Configuration
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
VECTOR_STORE_DIR = os.path.join(STORAGE_DIR, "vector_store")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.json")

# Create storage directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# config.py
# config.py
PROMPTS = {
    "agent": {
        "ibm_expert": {
            "role": "IBM Error Code Expert",
            "goal": "Find and explain IBM error codes by dynamically identifying key information from all sources",
            "backstory": "You're an expert in IBM error codes who intelligently analyzes documentation and web resources to extract the most relevant information, creating dynamic labels based on content.",
            "tools": ["search_vectorstore", "web_search", "format_content"]
        }
    },
    "task": {
        "research": """Analyze this IBM error code from all available sources and identify key information:

Query: {query}

Context from documents:
{context}

Conversation history:
{chat_history}

Instructions:
1. Must provide the output for asked code only.
2. Dynamically create labels based on content
3. Combine information from documents and web under matching labels
4. Summarize when multiple sources provide similar information
5. Keep response concise (max 15 lines total)
6. Mark web-sourced information with (web)""",

        "expected_output": """
        Analyze the content and present information using these dynamic guidelines:

        Error Code: {query}
        Content Sources: {context}

        [Dynamically Identified Label 1]:
        - Consolidated information from all sources
        - Summary when similar info exists
        
        [Dynamically Identified Label 2]:
        - Merged details from documents and web
        - Web sources marked with (web)
        
        [Dynamically Identified Label 3]:
        - Additional relevant information
        - Keep concise

        Rules:
        1. Create labels based on content attributes (e.g., "Error Severity", "Affected Components")
        2. Only include labels with actual information
        3. Merge duplicate information under same label
        4. Maximum 3-10 most important labels
        5. MUST keep within 15 line limit for entire response
        6. Prioritize technical details from documentation
        7. Web supplements marked with (web)"""
    },
    "errors": {
        "not_found": """I couldn't find information about {error_code} in our documentation.

Please:
1. Verify the exact error code
2. Check IBM's official documentation
3. Contact support with:
   - Full error message
   - System logs
   - Steps to reproduce"""
    },
    "formatting": {
        "content_format": """Convert this technical content to clean format:
        {content}
        
        Requirements:
        - Maintain dynamic labeling
        - Merge similar information
        - Keep error codes intact
        - Remove redundant details
        - Mark web sources with (web)"""
    }
}