import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO
import docx 


tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

@st.cache_data
def setup_docs_from_pdf(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()  
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs

@st.cache_data
def setup_docs_from_word(word_file):
    # Read the Word file
    doc = docx.Document(word_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return full_text

def generate_text(prompt, model, tokenizer, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=model.config.n_positions)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    input_ids = inputs['input_ids']
    if input_ids.size(1) > model.config.n_positions:
        raise ValueError("The prompt is too long for the model's maximum token limit.")
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=inputs.get('attention_mask'),
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_chunks(chunks, model, tokenizer, max_new_tokens):
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following text:\n\n{chunk}"
        summary = generate_text(prompt, model, tokenizer, max_new_tokens)
        summaries.append(summary)
    return " ".join(summaries)

def main():
    st.set_page_config(layout="wide")
    st.title("Document Summarization App")

    # Sidebar
    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=500, max_value=5000, step=500, value=2000)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=50, max_value=1000, step=50, value=200)
    max_new_tokens = st.sidebar.number_input("Max Number of New Tokens for Summary", min_value=50, max_value=500, value=150)

    # Option to choose input
    input_method = st.selectbox("Select Input Method", ["Upload PDF", "Upload Word Document", "Enter Text"])

    if input_method == "Upload PDF":
        pdf_file = st.file_uploader("Upload PDF file", type="pdf")
        if pdf_file:
            # Save the PDF to a temporary file
            with open("temp_pdf.pdf", "wb") as f:
                f.write(pdf_file.read())
            pdf_file_path = "temp_pdf.pdf"
            docs = setup_docs_from_pdf(pdf_file_path, chunk_size, chunk_overlap)
            st.write(f"PDF Loaded Successfully with {len(docs)} chunks.")
        text_input = None

    elif input_method == "Upload Word Document":
        word_file = st.file_uploader("Upload Word Document", type="docx")
        if word_file:
            full_text = setup_docs_from_word(word_file)
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.create_documents(full_text)
            st.write(f"Word Document Loaded Successfully with {len(docs)} chunks.")
        text_input = None

    elif input_method == "Enter Text":
        text_input = st.text_area("Enter the text to summarize here:")
        docs = None
        if text_input:
            # Process the entered text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.create_documents([text_input])
            st.write(f"Text Input Loaded Successfully with {len(docs)} chunks.")

    if st.button("Summarize"):
        if docs or text_input:
            try:
                if text_input:
                    
                    full_text = text_input
                else:
                    
                    full_text = " ".join([doc.page_content for doc in docs])
                
                max_chunk_length = tokenizer.model_max_length - max_new_tokens
                chunks = [full_text[i:i + max_chunk_length] for i in range(0, len(full_text), max_chunk_length)]
                final_summary = summarize_chunks(chunks, model, tokenizer, max_new_tokens)
                st.write("Summary:")
                st.write(final_summary)
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please provide input via PDF, Word document, or text before attempting to summarize.")

if __name__ == "__main__":
    main()
