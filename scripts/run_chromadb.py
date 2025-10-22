
import os,sys
from pathlib import Path
import chromadb
from tqdm import tqdm
from datetime import datetime,date
from typing import List
# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.preprocess import convert_document_to_markdown, chunk_document
from utils.models import ChunkModel
from provider import get_embedding

# 1 convert_chunks_to_vector
# 2. add_vectors_to_db

# Create a new Chroma client with persistence enabled. 
chromadb_master_folder = "./chromadb"
client = chromadb.PersistentClient(path=chromadb_master_folder)
# Create a new chroma collection
collection_name = "peristed_collection"



def convert_chunks_to_vector(chunks:List[ChunkModel])->List[ChunkModel]:
    if not isinstance(chunks,list) :
        raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    if len(chunks) == 0 :
        raise Exception(f"Chunk to vector conversion does not accept empty list")
    
    if not all([isinstance(chunk,ChunkModel) for chunk in chunks]):
        raise TypeError("Chunk to vector conversion must start with input as list of ChunkModel")
    
    chunks_text = [chunk.page_content for chunk in chunks]
    try:
        vectors = get_embedding(chunks_text)
    except Exception as e:
        raise f"Cannot convert Chunks into vector with error: {e}"
    
    for vector,chunk in zip(vectors,chunks):
        chunk.vector = vector
        
    return chunk

def add_vectors_to_db(chunks: List[ChunkModel],collection_name:str)->None:
    
    collection = client.get_or_create_collection(name=collection_name)
    
    if not isinstance(chunks,list):
        raise TypeError(f"add_vectors_to_db function only accepts input as list of ChunkModel")
    
    if len(chunks) == 0:
        raise Exception(f"add_vectors_to_db function cannot accept empty list")
    
    if not any([isinstance(chunk,ChunkModel) for chunk in chunks]):
         raise TypeError(f"add_vectors_to_db function only accepts input as list of ChunkModel")    
     
    collection.add(
                embeddings=[chunk.vector for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                documents=[chunk.page_content for chunk in chunks],
            ids=[chunk.metadata['index'] for chunk in chunks],)

if __name__ == "__main__":
    
    doc_master_folder = "documents"
    markdown_master_folder = "markdown_docs"
    doc_filename = "E_BOCHK_AR.pdf"
    doc_filepath = os.path.join(doc_master_folder,doc_filename)
    markdown_doc_filename = Path(doc_filepath).stem+".md"
    markdown_doc_filepath = Path(markdown_master_folder) / markdown_doc_filename
    raw_docs = convert_document_to_markdown(doc_filepath)
    chunks = chunk_document(raw_docs)
    chunks_with_vectors = convert_chunks_to_vector(chunks)



# print("adding documents and embeddings into docu")
# # Add some data to the collection


# text = "What is the ETF performance?"
# text_emb = embeddings(text)
# results = collection.query(
#     query_embeddings=text_emb,
#     n_results=5)

# print(results)