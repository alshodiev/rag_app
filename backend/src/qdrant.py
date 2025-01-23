from langchain.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import CollectionInfo

import sys
sys.path.insert(0, '/Users/alshodiev/opt/anaconda3/envs/rag_app/lib/python3.10/site-packages')

from decouple import config

qdrant_api_key = config('QDRANT_API_KEY')
qdrant_api_url = config('QDRANT_API_URL')
collection_name = "websites"

client = QdrantClient(url = qdrant_api_url, 
                      api_key = qdrant_api_key)

vector_store = Qdrant(
    client = client,
    collection_name = collection_name,
    embeddings = OpenAIEmbeddings(
        api_key = config('OPENAI_API_KEY')
    )
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                               chunk_overlap = 20, 
                                               length_function = len)

def create_collection(collection_name):
    client.create_collection(collection_name = collection_name, 
                             vectors_config = models.VectorParams(size = 1536, 
                                                                   distance = models.Distance.COSINE))
    print(f"Collection {collection_name} created")

def upload_website_to_collection(url:str):
    loader = WebBaseLoader(url)
    docs = loader.load_and_split(text_splitter)
    for doc in docs:
        doc.metadata = {"source_url" : url}
    
    vector_store.add_documents(docs)
    return f"Successfully uploaded {len(docs)} documents from {url}"

#client.delete_collection("websites")
#print("Collection 'websites' deleted.")

#create_collection(collection_name)
#upload_website_to_collection("https://hamel.dev/blog/posts/evals/")


    
