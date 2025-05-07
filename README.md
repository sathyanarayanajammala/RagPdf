# IBM Error Code PDF Chat

A RAG (Retrieval-Augmented Generation) application for chatting with IBM Error code PDF documents.

## Features

- Upload and process IBM Error code PDF documents
- Chat with the uploaded documents using natural language
- Dark mode UI with Gradio
- Fast PDF processing for better user experience
- FAISS vector database for efficient retrieval
- Multiple agents for comprehensive responses:
  - Research Agent: Analyzes PDF content
  - Web Search Agent: Finds supplementary information
  - Formatter Agent: Formats responses for clarity

## Technologies Used

- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Google's embedding model (compatible with Gemini)
- **Vector Database**: FAISS
- **UI**: Gradio
- **PDF Processing**: PyPDF
- **Agent Framework**: CrewAI

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Run the application:
   ```
   python app.py
   ```

## Usage

1. Upload IBM Error code PDF documents using the file upload button
2. Ask questions about the IBM Error codes in the chat interface
3. View responses from the AI agents

## Project Structure

- `app.py`: Main application file with Gradio UI
- `pdf_processor.py`: PDF loading and processing
- `vector_db.py`: FAISS vector database management
- `agents.py`: Implementation of AI agents using CrewAI
