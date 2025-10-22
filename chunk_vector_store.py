

from typing import List
import chromadb
from utils.preprocess import convert_document_to_markdown,chunk_document
from utils.preprocess import convert_document_to_markdown, chunk_document
from utils.models import ChunkModel
from provider import get_embedding

# Create a new Chroma client with persistence enabled. 
chromadb_master_folder = "./chromadb"
client = chromadb.PersistentClient(path=chromadb_master_folder)

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
        
    return chunks

def add_vectors_to_db(collection_name:str,chunks: List[ChunkModel],)->None:
    
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


class ChunkVectorStore:

  def __init__(self) -> None:
    pass

  def split_into_chunks(self, file_path: str):
    raw_docs = convert_document_to_markdown(file_path)
    chunks = chunk_document(raw_docs)

    return chunks

  def store_to_vector_database(self, collection_name, chunks):
    chunks_with_vectors = convert_chunks_to_vector(chunks)
    add_vectors_to_db(collection_name, chunks_with_vectors)