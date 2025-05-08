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
PROMPTS = {
    "agent": {
        "ibm_expert": {
            "role": "IBM Error Code Expert",
            "goal": "Find and explain IBM error codes accurately",
            "backstory": "You're an expert in IBM error codes...",
            "tools": ["search_vectorstore",  "format_content"]
        }
    },
    "task": {
        "research": """Research this IBM error code:

Query: {query}

Context from documents:
{context}

Conversation history:
{chat_history}

Provide:
""",
    "expected_output": """
    Analyze this IBM error code documentation and extract information using these dynamic labels:
    
    Error Code: {query}
    Document Content: {context}
    
    Extract and present the following information (if available):
    
    [Dynamic Label 1]: (Automatically identify the most relevant attribute from the content)
    [Response for Dynamic Label 1]
    
    [Dynamic Label 2]: (Automatically identify the next relevant attribute)
    [Response for Dynamic Label 2]
    
    [Dynamic Label 3]: (Continue with additional attributes as needed)
    [Response for Dynamic Label 3]
    
    Guidelines:
    1. Create clear labels based on content (e.g., "Error Severity", "Affected Components")
    2. Only include labels for which you find information
    3. Prioritize technical details from the documentation
    """
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
        "content_format": """Convert this technical content to clean plain text:
        {content}
        
        Requirements:
        - Remove markdown formatting
        - Keep error codes as-is (e.g. DSNB350I)
        - Use simple bullet points
        - Remove special characters except dashes"""
    }

}