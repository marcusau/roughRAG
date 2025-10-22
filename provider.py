import os,sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime,date
from typing import List,Union
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_xai import ChatXAI

# Add the project root to Python path


from utils.preprocess import timer

# Load environment variables from .env file
load_dotenv()
os.environ["XAI_API_KEY"] = os.environ.get("XAI_API_KEY")


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

def get_embedding(text:Union[str,List[str]])->List[float]:
    if isinstance(text,str) or isinstance(text,list) or all(isinstance(item,str) for item in text):
        embedding_func = EmbeddingFunction(OllamaEmbeddings( model="nomic-embed-text:latest"))
        return embedding_func(text)
    else:
        raise TypeError(f"text must be a string or a list of strings")
    
def get_llm()->ChatXAI:
    model= ChatXAI(
    model="grok-3-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
    return model