import os
from pathlib import Path
import pymupdf4llm
from docling.document_converter import DocumentConverter

doc_master_folder = "documents"
markdown_master_folder = "markdown_docs"
if not os.path.exists(markdown_master_folder):
    os.makedirs(markdown_master_folder)
doc_filename = "E_BOCHK_AR.pdf"
doc_filepath = os.path.join(doc_master_folder,doc_filename)
markdown_doc_filename = Path(doc_filepath).stem+".md"
markdown_doc_filepath = Path(markdown_master_folder) / markdown_doc_filename

# md_text = pymupdf4llm.to_markdown(doc_filepath,show_progress=True)

# for md in md_text:
#     if isinstance(md,dict):
#         metadata = md['metadata']
#         file_path = metadata['file_path']
#         page = metadata['page']
#         text = md['text']
#         print(f"File path: {file_path}, Page: {page}, Text: {text}")
#         print("-"*100)

# markdown_doc_filepath.write_bytes(md_text.encode())

