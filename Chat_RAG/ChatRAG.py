import os
import google.generativeai as genai
import warnings
import sys
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

# Suppress LangChain and Chroma warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Ensure this is set in your environment
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# Paths and embedding function
CHROMA_PATH = "chroma"
em_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Query (dynamic via command-line argument or default)
query_text = sys.argv[1] if len(sys.argv) > 1 else "Where is my resume?"

# Load Chroma database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=em_func)

# Retrieve top 3 relevant chunks (use similarity_search to avoid relevance score warning)
result = db.similarity_search(query_text, k=3)

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {query}
"""

# Create context from retrieved chunks
context_text = "\n\n---\n\n".join([doc.page_content for doc in result])

# Format prompt
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format_messages(context=context_text, query=query_text)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Generate response using Gemini API
try:
    response = model.generate_content(prompt[0].content)
    print(response.text)
except Exception as e:
    print(f"Error generating response with Gemini: {str(e)}")