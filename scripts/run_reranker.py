
import uuid
import os,sys
import chromadb

from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime,date
from typing import List,Union,Dict,Any

from langchain_core.prompts import PromptTemplate
from langchain_xai import ChatXAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from provider import get_llm, get_embedding
from utils.models import RerankedResponsesModel,DBSearchModel
from prompt import Prompts



# 1 get llm
# 2 db_search_func
# 3 rerank_func
# 4 rag_generator_func

prompts = Prompts()
model = get_llm()


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
        raise f"Fail to generate re-ranking response from model as error :{e}"
    


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
        raise f"Fail to generate response from RAG Generator model with error : {e}"

# Create a new Chroma client with persistence enabled. 
chromadb_master_folder = "./chromadb"
client = chromadb.PersistentClient(path=chromadb_master_folder)

if __name__ == "__main__":
    # Create a new chroma collection
    collection_name = "peristed_collection"
    collection = client.get_or_create_collection(name=collection_name)



    query_text = "What is the ETF performance?"
    search_results=db_search_func(collection,query_text)
    reranking_results = rerank_func(query_text,chunks=search_results)
    # print("Reranked results:")
    # for label in reranking_results.labels:
    #     print(f"Chunk {label.chunk_id} (Relevancy: {label.relevancy}):")
    #     print(f"Text: {label.text}")
    #     #print(f"Text: {chunks[label.chunk_id]['text']}")
    #     print(f"Reasoning: {label.chain_of_thought}")
    #     print()

    rag_response=rag_generator_func(query=query_text, chunks=reranking_results)
    print(f"RAG Response:\n {rag_response.content}")
    

