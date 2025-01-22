from sentence_transformers import SentenceTransformer
import faiss
import os
from transformers import pipeline

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare your data
documents = []
folder_path = "Knowledge_Base"
for file in os.listdir(folder_path):
    with open(os.path.join(folder_path, file), 'r') as f:
        documents.append(f.read())

# Generate embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index for reuse (optional)
faiss.write_index(index, "local_data.index")

# Query the data
query = input("Prompt : ")
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding, k=5)  # Retrieve top-5 matches

# Get the corresponding documents
relevant_docs = [documents[i] for i in indices[0]]

# Load a pre-trained language model for response generation
# generator = pipeline("text-generation", model="gpt-neo-2.7B")
generator = pipeline("text-generation", model="gpt2",  trust_remote_code=True)
# Combine query with retrieved data and generate a response
context = "\n".join(relevant_docs)
input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
response = generator(input_text, max_new_token=150, num_return_sequences=1)
print("Generated Response:")
print(response[0]['generated_text'])
