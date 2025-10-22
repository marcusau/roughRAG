import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime,date
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

doc_master_folder = "documents"
markdown_master_folder = "markdown_docs"
doc_filename = "E_BOCHK_AR.pdf"
doc_filepath = os.path.join(doc_master_folder,doc_filename)
markdown_doc_filename = Path(doc_filepath).stem+".md"
markdown_doc_filepath = Path(markdown_master_folder) / markdown_doc_filename

# with open(markdown_doc_filepath,"r") as f:
#     docs = f.read()
md_text = pymupdf4llm.to_markdown(doc_filepath,show_progress=True)


# md_text = pymupdf4llm.to_markdown(doc_filepath,show_progress=True,page_chunks=True)

# docs=[]
# for md in tqdm(md_text,total=len(md_text),desc="split pages and metadata into Langchain Documents"):
#     if isinstance(md,dict):
#         metadata = md['metadata']
#         file_path = metadata['file_path']
#         page = metadata['page']
#         text = md['text']
#         doc = Document(page_content=text,metadata={'source':file_path,"page":page,"create_date": date.today()})
#         docs.append(docs)
        #print(f"File path: {file_path}, Page: {page}, Text: {text}")
        #print("-"*100)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    #separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " ", ""]
) 

docs = text_splitter.split_text(md_text)
print(len(docs))


