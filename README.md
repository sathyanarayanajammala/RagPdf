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

##  Error outputs:

##  Error 1
**IBM Db2 Error Code 00C20031

Explanation:

IBM Db2 error code 00C20031 indicates that a resource is not available. Specifically, it is often associated with a page latch timeout problem.

System Action:

When this error occurs, the requesting execution unit is abended to take a diagnostic dump. It is then recovered to return an SQLCODE -904 (resource not available) to the user, along with the 00C20031 reason code. The resource name is also provided.

User Response:

Contact the system programmer to determine why the resource is unavailable.
Print the SYS1.LOGREC.
Print the SVC dump.
Problem Determination:

This dump provides assistance for diagnosing the page latch timeout problem. Further investigation by the system programmer is needed to determine the root cause of the resource unavailability.

Possible causes and further actions:

Page Latch Timeout: This suggests contention for a specific page in the database. Analyze the dump to identify the page and the processes involved.
Resource Unavailability: The underlying resource (table space, index) might be offline or in an inconsistent state. Check the Db2 logs and system status to verify the resource's availability.
Internal Db2 Error: While less common, this could point to an internal Db2 issue. If other troubleshooting steps fail, consider contacting IBM support.
Related Error Codes:

SQLCODE -904: This SQLCODE is returned to the user in conjunction with the 00C20031 reason code, indicating a resource is unavailable.
Note:

The 00C2010F dump provides assistance for diagnosing the page latch timeout problem. The 00C20031 reason code is returned as 'resource not available'.

## Error 2:

**IBM Db2 Error Code 00C200A3

Explanation:

This is a Db2/MVS internal error. The execution unit driving a buffer manager (BM) asynchronous function, which would normally run indefinitely, has been canceled. However, work being done by the execution unit is allowed to complete before the execution unit terminates.

System Action:

The affected asynchronous function is terminated. If the affected function is the deferred write processor (DSNB1CMS), Db2 is abended with the abend code '00C200D3'.

Operator Response:

Notify the system programmer.
Print the SYS1.LOGREC.
Request the SVC dump.
Start Db2 if it is abended.
System Programmer Response:

If you suspect an error in Db2, you might need to report the problem. See "Collecting diagnostic data (Collecting data)" for information about identifying and reporting the problem.

Problem Determination:

Dynamic dump, taken to SYS1.DUMPxx data set, by Db2 (04E and 04F abends).
A listing of the SYS1.LOGREC data set, obtained by executing IFCEREP1.

Screen shots:

