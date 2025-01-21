from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import os
from transformers import pipeline

app = Flask(__name__)

# Initialize models and index globally
model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text-generation", model="gpt2", trust_remote_code=True)

# Load FAISS index and documents
def load_data():
    documents = []
    folder_path = "Knowledge_Base"
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r') as f:
            documents.append(f.read())
    
    index = faiss.read_index("local_data.index")
    return documents, index

documents, index = load_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    query = request.json.get('query', '')
    
    # Generate embeddings and search
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)
    
    # Get relevant documents
    relevant_docs = [documents[i] for i in indices[0]]
    
    # Generate response
    context = "\n".join(relevant_docs)
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = generator(input_text, max_length=300, num_return_sequences=1)
    
    return jsonify({
        'response': response[0]['generated_text'],
        'relevant_docs': relevant_docs[:3]  # Return top 3 relevant documents
    })

if __name__ == '__main__':
    app.run(debug=True)