from crewai.tools import tool as CrewAITool
from langchain_community.tools import DuckDuckGoSearchRun
import logging
from config import PROMPTS

logger = logging.getLogger('ibm_error_code_rag')

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

@CrewAITool
def search_vectorstore(query: str) -> str:
    """Search the vector store for relevant documents about IBM error codes."""
    from agents import vector_store  # Import here to avoid circular imports
    
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
        logger.error(f"Error searching vector store: {str(e)}")
        return f"Error searching vector store: {str(e)}"

@CrewAITool
def web_search(query: str) -> str:
    """Search IBM forums, documentation, and Mainframe communities for error code solutions."""
    try:
        # Target IBM's official sites and communities
        sites = [
            "site:ibm.com",
            "site:community.ibm.com",
            "site:stackoverflow.com"
        ]
        site_filters = " OR ".join(sites)
        search_query = f"IBM error code {query} ({site_filters})"
        
        result = search_tool.run(search_query)
        return result if result else "No results found in IBM/Mainframe sources."
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return f"Search failed: {str(e)}"

@CrewAITool
def format_content(content: str) -> str:
    """Format technical content about IBM error codes for better readability."""
    try:
        from agents import llm  # Import here to avoid circular imports
        formatted = llm.invoke(
           PROMPTS["formatting"]["content_format"].format(content=content)
        )
        return formatted.content
    except Exception as e:
        logger.error(f"Error formatting content: {str(e)}")
        return f"Error formatting content: {str(e)}\n\nOriginal content:\n{content}"