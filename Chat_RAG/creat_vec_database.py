import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Paths
DATA_PATH = "D:\\projects\\Personal-Agent\\folder_memory.json"  # Replace with your JSON file path, e.g., "data.json"
CHROMA_PATH = "chroma"  # Adjusted to a relative path for portability

def load_json_data(json_path):
    """Load JSON data from the specified path."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def json_to_documents(json_data):
    """Convert JSON data into LangChain Document objects."""
    documents = []
    
    for folder_path, folder_info in json_data.items():
        # Extract folder-level metadata
        folder_description = folder_info.get('description', '')
        folder_summary = folder_info.get('summary', {})
        timestamp = folder_info.get('timestamp', '')
        file_paths = folder_info.get('file_paths', {})
        details = folder_info.get('details', [])
        
        # Create content string for embedding
        summary_text = (
            f"Total Files: {folder_summary.get('total_files', 0)}\n"
            f"Total Folders: {folder_summary.get('total_folders', 0)}\n"
            f"File Types: {', '.join([f'{k}: {v}' for k, v in folder_summary.get('file_types', {}).items()])}\n"
            f"Largest File: {folder_summary.get('largest_file', '')} ({folder_summary.get('largest_size', 0)} bytes)"
        )
        file_paths_text = '\n'.join([f"{name}: {path}" for name, path in file_paths.items()])
        details_text = '\n'.join(details)
        
        content = (
            f"Folder Path: {folder_path}\n"
            f"Description: {folder_description}\n"
            f"Summary:\n{summary_text}\n"
            f"Files:\n{details_text}\n"
            f"File Paths:\n{file_paths_text}"
        )
        
        # Create metadata
        metadata = {
            'folder_path': folder_path,
            'timestamp': timestamp,
            'total_files': folder_summary.get('total_files', 0),
            'total_folders': folder_summary.get('total_folders', 0),
            'largest_file': folder_summary.get('largest_file', ''),
            'largest_size': folder_summary.get('largest_size', 0),
            'file_paths': json.dumps(file_paths)  # Serialize dictionary to string
        }
        
        # Create LangChain Document
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def split_text(documents):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vec_db(chroma_path, chunks):
    """Create and persist a Chroma vector database from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    db.persist()  # Save the database to disk
    return db

def main():
    # Load JSON data
    json_data = load_json_data(DATA_PATH)
    
    # Convert JSON to documents
    documents = json_to_documents(json_data)
    
    # Split documents into chunks
    chunks = split_text(documents)
    
    # Create vector database
    db = create_vec_db(CHROMA_PATH, chunks)
    
    print(f"Vector database created at {CHROMA_PATH} with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()