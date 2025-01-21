from sentence_transformers import SentenceTransformer
import faiss
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare your data
documents = []
folder_path = "D:\projects\RAG\Knowledge_Base"
for file in os.listdir(folder_path):
    with open(os.path.join(folder_path, file), 'r') as f:
        documents.append(f.read())

# Generate embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index for reuse
faiss.write_index(index, "local_data.index")
