from crewai import Agent, Task
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from config import VECTOR_STORE_DIR, METADATA_FILE,STORAGE_DIR
import json
import logging
from tools import search_vectorstore, web_search, format_content  # Import tools here
from config import llm, PROMPTS   

logger = logging.getLogger('ibm_error_code_rag')

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


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
    
    expected_output = """
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
        """.format(query=query, context=context or "No relevant documents found")
    
    return Task(
        description=description,
        agent=agent,
        expected_output=expected_output
    )


