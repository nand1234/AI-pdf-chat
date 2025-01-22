from langchain_community.document_loaders import WebBaseLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document  # Adjust import based on your library
import tempfile
import io



DB_dir: str = f"./langchain/"

def load_documents(url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/"):
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(data)

    local_embidding = HuggingFaceEmbeddings()

    Chroma.from_documents(documents=all_splits, embedding=local_embidding, persist_directory=DB_dir)
    


def load_pdf(pdf):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf.read())
        temp_file_path = temp_file.name  # Get the temporary file path
    loader = PDFMinerLoader(temp_file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(data)

    local_embidding = HuggingFaceEmbeddings()

    Chroma.from_documents(documents=all_splits, embedding=local_embidding, persist_directory=DB_dir)

def get_context(question):
    local_embidding = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=DB_dir, embedding_function=local_embidding)
    docs = vectorstore.similarity_search(question, k=3)
    combined_page_content = ""
    combined_metadata = {}
    for doc in docs:
        # Safely retrieve and concatenate page_content
        if hasattr(doc, 'page_content') and doc.page_content:
            combined_page_content += doc.page_content + "\n\n"
        
        # Safely merge metadata
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            combined_metadata.update(doc.metadata)

    # Remove trailing newlines
    combined_page_content = combined_page_content.strip()

    # Create and return a new Document
    return Document(
        page_content=combined_page_content,
        metadata=combined_metadata
    )

if __name__ == '__main__':
    uploaded_file = 'sample_pdf/Nandkumar_Ghatage_Latest_CV.pdf'
    # Open the file in binary mode and read its content
    with open(uploaded_file, 'rb') as f:
         pdf_data = f.read()
    # Convert byte data to a file-like object using io.BytesIO
    pdf_file = io.BytesIO(pdf_data)
    
    # Extract text from the uploaded PDF
    load_pdf(pdf_file)
    doc = get_context('whats is this CV about?')
    print(doc)