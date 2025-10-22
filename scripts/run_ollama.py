from langchain_ollama import OllamaEmbeddings
import ollama

import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime,date
import pymupdf4llm

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class EmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, embeddings_obj):
        self. embeddings_obj =  embeddings_obj

    @timer
    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.embeddings_obj.embed_documents(input,)
    
embeddings = EmbeddingFunction(OllamaEmbeddings( model="nomic-embed-text:latest"))

doc_master_folder = "documents"
markdown_master_folder = "markdown_docs"
doc_filename = "E_BOCHK_AR.pdf"
doc_filepath = os.path.join(doc_master_folder,doc_filename)
markdown_doc_filename = Path(doc_filepath).stem+".md"
markdown_doc_filepath = Path(markdown_master_folder) / markdown_doc_filename

# with open(markdown_doc_filepath,"r") as f:
#     docs = f.read()
    
 
# md_text = pymupdf4llm.to_markdown(doc_filepath,show_progress=True,page_chunks=True)

# docs=[]
# for md in tqdm(md_text,total=len(md_text),desc="split pages and metadata into Langchain Documents"):
#     if isinstance(md,dict):
#         metadata = md['metadata']
#         file_path = metadata['file_path']
#         page = metadata['page']
#         text = md['text']
#         doc = Document(page_content=text,metadata={'source':file_path,"page":page,"create_date": date.today().strftime("%Y-%m-%d")})
#         docs.append(doc)


# doc_texts= [doc.page_content for doc in docs]

#md_text = pymupdf4llm.to_markdown(doc_filepath,show_progress=True,)   
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False,
#     separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " ", ""]
# ) 

# split_texts = text_splitter.split_text(md_text)

# docs=[]
# for idx, doc,emb in enumerate(split_texts):
#     source = str(Path(doc_filepath).stem).replace("/","_").replace(".","_").strip()
#     doc_idx = "_".join([source, str(idx)])
#     doc = Document(page_content=doc,metadata={"index":doc_idx,'source':str(Path(doc_filepath).stem),"create_date": date.today().strftime("%Y-%m-%d")})
#     docs.append(doc)
# print(f"numbers of text split: {len(docs)}")

# doc_texts= [doc.page_content for doc in docs]
# embs= embeddings(doc_texts)

