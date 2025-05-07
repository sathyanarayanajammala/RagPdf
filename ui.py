import gradio as gr
from config import STORAGE_DIR, VECTOR_STORE_DIR, METADATA_FILE
from agents import document_metadata
import logging

logger = logging.getLogger('ibm_error_code_rag')

def create_ui(process_files_fn, chat_fn, clear_vector_store_fn, refresh_logs_fn, clear_chat_and_logs_fn):
    """Create and return the Gradio UI interface"""
    with gr.Blocks(theme="gradio/default") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("** Upload IBM Error Code PDFs")
                file_input = gr.Files(label="Upload PDF Files", file_types=[".pdf"], file_count="multiple")
                upload_button = gr.Button("Process Files")
                clear_store_button = gr.Button("Clear Vector Store", variant="secondary")
                status_text = gr.Textbox(label="Processing Status", interactive=False)

                with gr.Row():
                    log_output = gr.Textbox(label="Processing Logs", interactive=False, lines=10, autoscroll=True)
                    refresh_logs = gr.Button("Refresh Logs")

                stats_output = gr.JSON(label="Processing Statistics", visible=True)
                indexed_docs_html = gr.HTML(label="Indexed Documents")

            with gr.Column(scale=2):
                gr.Markdown("** Chat with IBM Error Code Documents")
                chatbot = gr.Chatbot(height=400, type='tuples')
                message_input = gr.Textbox(label="Ask a question about IBM error codes", placeholder="Type your question here...")
                clear_button = gr.Button("Clear Chat")

        # Set up event handlers
        upload_button.click(
            process_files_fn, 
            inputs=[file_input], 
            outputs=[log_output, status_text, stats_output, indexed_docs_html]
        )
        clear_store_button.click(
            clear_vector_store_fn, 
            outputs=[log_output, status_text, stats_output, indexed_docs_html]
        )
        refresh_logs.click(refresh_logs_fn, outputs=log_output)

        message_input.submit(
            fn=chat_fn,
            inputs=[message_input, chatbot],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[message_input]
        )

        clear_button.click(
            clear_chat_and_logs_fn, 
            outputs=[chatbot, log_output]
        )

    return demo

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