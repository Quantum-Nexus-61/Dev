import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import base64
import os

# Model and tokenizer loading
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = [text.page_content for text in texts]
    return final_texts

# Text summarization function
def text_summarization(text, max_length=150, min_length=50):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_length, min_length=min_length, truncation=True)
    summary_ids = model.generate(input_ids)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Display PDF function
def displayPDF(file):
    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while displaying the PDF: {str(e)}")

# Main function
def main():
    # Set background color and page layout
    st.set_page_config(
        page_title="LEXI GENIUS || AN OFFLINE LLM BRILLANCE",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS to style the output
    st.markdown(
        """
        <style>
            .stApp {
                background-color: black;
            }
            .stMarkdown {
                color: white;
            }
            .stProgress > div > div {
                background-color: #4CAF50;
            }
            .stButton {
                background-color: #008CBA;
                color: white;
            }
            .stButton:hover {
                background-color: #005682;
            }
            .stSuccess {
                color: #4CAF50;
            }
            .stError {
                color: #FF0000;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main content
    col1, col2 = st.columns([1, 3])  # Adjusting column widths

    # Add a logo next to the title (reduced size)
    col1.image("https://www.shutterstock.com/image-vector/chatbot-icon-concept-chat-bot-600nw-2132342911.jpg", width=150)

    # Title with white color on a black background
    col2.title("LEXI GENIUS || AN OFFLINE LLM BRILLANCE")

    # Image on the left, text on the right
    col1.image("https://pdf-summarizer.com/images/PDF%20Summarizer%20(3).png", use_column_width=True)

    # Text on the right with white color
    col2.markdown(
        """
        Welcome to Lexi Genius! This tool allows you to upload a PDF file and generates a summary.
        
        Follow these simple steps:
        
        1. Upload your PDF file using the 'Upload your PDF file' button below.
        2. Adjust summarization parameters (if needed).
        3. Click the 'Generate Summary' button to generate a summary.
        """
    )

    uploaded_file = col2.file_uploader("Choose a PDF file to summarize", type=['pdf'])

    if uploaded_file is not None:
        col2.write(f"Selected File: {uploaded_file.name}")
        
        # Interactive parameters
        max_length = col2.slider("Maximum Summary Length", min_value=50, max_value=300, value=150)

        if col2.button("Generate Summary"):
            filepath = "data/" + uploaded_file.name

            os.makedirs("data", exist_ok=True)

            try:
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                with col2:
                    st.success(f"{uploaded_file.name} uploaded successfully!")
                    pdf_view = displayPDF(filepath)

                with col2:
                    progress_bar = st.progress(0)
                    with st.spinner("Summarizing..."):
                        input_texts = file_preprocessing(filepath)
                        total_steps = len(input_texts)

                        result = []
                        for i, text_chunk in enumerate(input_texts):
                            progress_bar.progress((i + 1) / total_steps)  # Update progress bar

                            chunk_summary = text_summarization(text_chunk, max_length=max_length)
                            result.append(chunk_summary)

                        summary = '\n'.join(result)
                    st.success("Summarization Complete")
                    st.markdown(summary, unsafe_allow_html=True)
            except Exception as e:
                col2.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
