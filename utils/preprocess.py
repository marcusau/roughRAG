import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union,List
from datetime import date
import functools


import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from functools import wraps

from utils.models import DocumentInfoModel,ChunkModel

# 1 timer
# 2 convert_document_to_markdown
# 3 chunk_document

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@functools.lru_cache(maxsize=3)
def read_txtfile(filepath:Union[str,Path])->str:
    if isinstance(filepath,str):
        filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")
    
    with open(filepath,'r',encoding='utf-8') as f:
        content = f.read()
    
    return content

@timer
def convert_document_to_markdown(filepath: Union[str, Path]) -> Tuple[str, Dict[str, Any]]:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"{filepath} does not exist")

    try:
        md_text = pymupdf4llm.to_markdown(filepath, show_progress=True)
        # Get document info
        doc_info = {
            "file": filepath.name,
            "extension": filepath.suffix,
            "file_path": str(filepath),
            "status": "Success",
            "length": len(md_text),
            "content": md_text,
        }
        return DocumentInfoModel(**doc_info)
    except Exception as e:
        raise f"Error converting document to markdown: {e}"


def chunk_document(raw_doc:DocumentInfoModel)-> List[ChunkModel]:
    
    if not isinstance(raw_doc,DocumentInfoModel):
        raise TypeError(f"raw document must be in type of DocumentInfoModel")
    if not isinstance(raw_doc.content, str):
        raise TypeError(f"{raw_doc.content} is not string. Only string is accept")
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        # separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " ", ""]
    )
    try:
        split_texts = text_splitter.split_text(raw_doc.content)

        docs = []
        for idx, doc in enumerate(split_texts):
            file_path_str = str(raw_doc.file.replace("/", "_").replace(".", "_").strip()) #str(Path(doc_filepath).stem).replace("/", "_").replace(".", "_").strip()
            doc_idx = "_".join([file_path_str, str(idx)])
            doc = ChunkModel(
                page_content=doc,
                metadata={
                    "index": doc_idx,
                    "source": raw_doc.file_path,
                    "create_date": date.today().strftime("%Y-%m-%d"),
                },
            )
            docs.append(doc)
        print(f"numbers of text split: {len(docs)}")

        return docs
    except Exception as e:
        raise f"Error when split documents into chunks - {e}"


if __name__ == "__main__":
    doc_master_folder = "documents"
    markdown_master_folder = "markdown_docs"
    if not os.path.exists(markdown_master_folder):
        os.makedirs(markdown_master_folder)
    doc_filename = "E_BOCHK_AR.pdf"
    doc_filepath = os.path.join(doc_master_folder, doc_filename)
    markdown_doc_filename = Path(doc_filepath).stem + ".md"
    markdown_doc_filepath = Path(markdown_master_folder) / markdown_doc_filename
    doc = convert_document_to_markdown(doc_filepath)
    print(doc)
