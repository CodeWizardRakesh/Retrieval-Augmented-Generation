import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# # === SETUP ===
# os.environ["GOOGLE_API_KEY"] = "your-api-key-here"  # Replace with your actual key

# === STEP 1: Scan Directories and Create Document Objects ===
def scan_directories(base_path):
    docs = []
    for root, dirs, files in os.walk(base_path):
        for name in files:
            path = os.path.join(root, name)
            text = f"File: {name} Path: {path}"
            docs.append(Document(page_content=text))
    return docs

# === STEP 2: Load & Embed ===
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# === STEP 3: Query Using Retrieval ===
def run_query(vectorstore, query):
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa({"query": query})
    return result['result']

# === MAIN ===
if __name__ == "__main__":
    directory_to_scan = input("Enter the base directory to scan files: ").strip()
    
    print("\n[1] Scanning directory and building vector store...")
    docs = scan_directories(directory_to_scan)
    vectorstore = create_vectorstore(docs)

    print("[2] Ready! You can now ask where files are located.\n")

    while True:
        user_query = input("Ask a file-related question (or type 'exit'): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = run_query(vectorstore, user_query)
        print(f"\nAnswer: {answer}\n")
