import uuid
from typing import List, Union,Dict
import chromadb

from langchain_core.prompts import PromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage

from chunk_vector_store import ChunkVectorStore 

from provider import get_llm, get_embedding
from utils.models import RerankedResponsesModel,DBSearchModel
from prompt import Prompts

prompts = Prompts()
# Create a new Chroma client with persistence enabled. 
chromadb_master_folder = "./chromadb"
client = chromadb.PersistentClient(path=chromadb_master_folder)




def db_search_func(collection,query:Union[str,List[str]],topk:int=10)->List[DBSearchModel]:
    
    if not isinstance(query,str) and not isinstance(query,list) or not all(isinstance(item,str) for item in query):
        raise TypeError(f"query must be a string or a list of strings")
    
    query_emb = get_embedding(query)
    results = collection.query(query_embeddings=query_emb, n_results=topk, )

    if results is None or len(results)==0:
        raise Exception(f"No result searched from vector DB with query: {query}")

    retrived_docs=results['documents']
    retrived_metadata=results['metadatas']
    retrived_distances=results['distances']
    search_items=[]
    for q_idx, q in enumerate(query if isinstance(query,list) else [query]):
        for idx,(text,meta,dist) in enumerate(zip(retrived_docs[q_idx],retrived_metadata[q_idx],retrived_distances[q_idx])):
            search_item = {"id":idx,"text":text,"metadata":meta,"distance":dist}
            search_items.append(DBSearchModel(**search_item))
    return search_items


 
def rerank_func(model,query:str,chunks: List[DBSearchModel]) -> RerankedResponsesModel:
    
    if not isinstance(chunks,list):
        raise TypeError("re-ranking function only accepts input type as list of DBSearchModel")
    
    if len(chunks) == 0:
        raise Exception("No VectorDB search item is provided")
    
    if not any([isinstance(chunk,DBSearchModel) for chunk in chunks]):
        raise TypeError("re-ranking function only accepts input type as list of DBSearchModel")
   
    # Format the chunks for the prompt
    chunks_text = ""
    for chunk in chunks:
        chunks_text += f'<chunk id="{chunk.id}">\n    {chunk.text}\n</chunk>\n'
    
    # Create the full prompt
    system_prompt=SystemMessage(content=prompts.rerank.system )
    
    human_prompt_template = PromptTemplate.from_template(template=prompts.rerank.user)
    human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
    human_prompt=HumanMessage(content=human_prompt_template)
    
    structured_model = model.with_structured_output(RerankedResponsesModel)
    try:
        resp=structured_model.invoke([system_prompt,human_prompt])
        return resp
    except Exception as e:
        raise Exception(f"Fail to generate re-ranking response from model as error :{e}")
    

def rag_generator_func(model,query: str, chunks: RerankedResponsesModel,) -> str:
    
    if not isinstance(chunks, RerankedResponsesModel):
        raise ValueError("chunks must be a RerankedResponsesModel")
    
    if not isinstance(query,str) or query.strip() in [None,""]:
        raise ValueError("query must be string and not empty string")
    
    chunks_text = ""
    for label in chunks.labels:
        chunks_text += f"Chunk id:{label.chunk_id}\nRelevancy: {label.relevancy}):\nReasoning:{label.chain_of_thought}\n Text: {label.text}\n\n\n"
    # Create the full prompt
    system_prompt=SystemMessage(content=prompts.generator.system)
    
    human_prompt_template = PromptTemplate.from_template(template=prompts.generator.user)
    human_prompt_template = human_prompt_template.format(query=query, chunks_text=chunks_text)
    human_prompt=HumanMessage(content=human_prompt_template)
    
    try:
        resp = model.invoke([system_prompt,human_prompt])
        return resp
    except Exception as e:
        raise Exception(f"Fail to generate response from RAG Generator model with error : {e}")


class RAG:

    vector_store = None
    retriever = None
    chain = None
  
    def __init__(self) -> None:
        self.csv_obj = ChunkVectorStore()
        self.collection_name = str(uuid.uuid4())  
        self.model = get_llm()
        
    def set_retriever(self):
        self.retriever = client.get_or_create_collection(name=self.collection_name)
        
      # Stores the file into vector database.
    def feed(self, file_path: str):
        # Ensure we have a valid collection name (may be cleared by previous sessions)
        if not self.collection_name:
            self.collection_name = str(uuid.uuid4())
        chunks = self.csv_obj.split_into_chunks(file_path)
        self.csv_obj.store_to_vector_database(self.collection_name,chunks)
        self.set_retriever()

    # Augment the context to original prompt.
    def augment(self,query:str):
        search_results=db_search_func(self.retriever,query)
        reranking_results = rerank_func(self.model, query, chunks=search_results)
        rag_response=rag_generator_func(self.model, query=query, chunks=reranking_results)
        return rag_response.content
    
    # Main method to ask questions to the RAG system
    def ask(self, query: str) -> str:
        if self.retriever is None:
            return "Please upload a document first before asking questions."
        
        if not query or query.strip() == "":
            return "Please provide a valid question."
        
        try:
            response = self.augment(query)
            return response
        except Exception as e:
            return f"Error processing your question: {str(e)}"
    
    # Delete the collection from ChromaDB
    def delete_collection(self):
        """Delete the current collection from ChromaDB"""
        try:
            if self.collection_name:
                client.delete_collection(name=self.collection_name)
                print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection {self.collection_name}: {str(e)}")
    
    # Get collection info
    def get_collection_info(self) -> Dict:
        """Get information about the current collection"""
        if self.retriever is None:
            return {"collection_name": self.collection_name, "document_count": 0}
        
        try:
            count = self.retriever.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count
            }
        except Exception as e:
            return {"collection_name": self.collection_name, "document_count": 0, "error": str(e)}
    
    def clear(self):
        """Clear the current session state but keep the existing collection name"""
        self.vector_store = None
        self.retriever = None