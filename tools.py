from crewai.tools import tool as CrewAITool
from langchain_community.tools import DuckDuckGoSearchRun
import logging

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
    """Search the web for information about IBM error codes."""
    try:
        return search_tool.run(f"IBM error code {query}")
    except Exception as e:
        logger.error(f"Error searching the web: {str(e)}")
        return f"Error searching the web: {str(e)}"

@CrewAITool
def format_content(content: str) -> str:
    """Format technical content about IBM error codes for better readability."""
    try:
        from agents import llm  # Import here to avoid circular imports
        formatted = llm.invoke(
            f"""Format the following technical content about IBM error codes for better readability.
            Use proper markdown with headings, lists, and code blocks where appropriate:
            
            {content}"""
        )
        return formatted.content
    except Exception as e:
        logger.error(f"Error formatting content: {str(e)}")
        return f"Error formatting content: {str(e)}\n\nOriginal content:\n{content}"