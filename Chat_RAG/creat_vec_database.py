from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

DATA_PATH = "/content/books"

def load_document():
  loader = DirectoryLoader(DATA_PATH)
  documents = loader.load()
  print(type(documents))
  return documents

def split_text(DATA_PATH):

  loader = DirectoryLoader(DATA_PATH)
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,# what function can be used to measure the length of the text during split
    add_start_index = True,# start indedx is metadata field
  )

  chunks = text_splitter.split_documents(documents)
  return chunks

def create_vec_db(CHROMA_PATH, chunks):

  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

  db = Chroma.from_documents(
      chunks,
      embedding = embeddings,
      persist_directory=CHROMA_PATH

  )
  
  
create_vec_db("/content/chroma", chunks=split_text(DATA_PATH))


# em_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vec = em_func.embed_query("Apple fdsfds")

# query_text = "What is the significance of the key in the story set in Elderglow?"

# db = Chroma(persist_directory="/content/chroma", embedding_function=em_func)
# result = db.similarity_search_with_relevance_scores(query_text, k = 3) #returns the best match of chunks