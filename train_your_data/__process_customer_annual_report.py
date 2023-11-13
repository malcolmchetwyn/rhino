import os
import fnmatch
import re
from langchain.embeddings import (OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings)
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document

directory_path = "./annual_reports"
#directory_path_test = "./annual_reports_test"
all_documents = []
embed_fn = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def extract_text_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.pdf':
        print(f"Extracting content from PDF file: {file_path}")
        with open(file_path, "rb") as pdf_file:
            pdf = PdfReader(pdf_file)
            return ''.join(page.extract_text() for page in pdf.pages).strip()

    elif file_extension == '.txt':
        print(f"Extracting content from TXT file: {file_path}")
        with open(file_path, "r") as file:
            return file.read().strip()

    elif file_extension == '.docx':
        print(f"Extracting content from DOCX file: {file_path}")
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def add_documents(text_content):
    print("Loading documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, 
                                                   separators=["\n\n", "\n", ".", ";", ",", " ", ""])
                                                       
    texts = text_splitter.split_text(text_content)
    all_documents.extend(texts)






def clean_pdf_text(text):
    # Split the data into lines and remove redundant spaces
    lines = [line.strip() for line in text.split("\n") if line.strip() != ""]

    enhanced_text = []

    for idx, line in enumerate(lines):
        # Financial year data followed by metrics, e.g. "5% on FY22$10,164m"
        if re.match(r"^\d+% on FY\d+\$\d+m", line):
            split_line = re.split(r"(\$)", line, 1)
            enhanced_text.append(split_line[0] + split_line[1] + split_line[2])
            enhanced_text.append("\n")  # Add an extra line for clarity

        # Financial metrics followed by details  
        elif re.match(r"\$\d+\.?\d*m", line) or re.match(r"\d+\.?\d*%", line):
            enhanced_text.append(line)
            if (idx + 1 < len(lines)):
                enhanced_text.append(lines[idx + 1])
            enhanced_text.append("\n")  # Add an extra line for clarity

        # Titles and sub-titles
        elif line in ["Balance Sheet", "Statement of Comprehensive Income", "RACV Annual Report",
                      "Investments Accounted for Using the Equity Method", "Balance Sheet", "COMMONWEAL TH BANK"]:
            enhanced_text.append("\n\n" + line + "\n" + "-"*len(line) + "\n")

        # Bullet points (e.g., • Arevo Pty Ltd)
        elif re.match(r"•\s", line):
            enhanced_text.append("  - " + line.split('•')[1])

        # Special notes
        elif line.startswith("1  Refer to note"):
            enhanced_text.append("\nNote: " + line + "\n")

        # Categories followed by descriptions (e.g., "Strategic", "Financial")
        elif line in ["Strategic", "Financial", "Non ‑financial", "EmergingRisk", "Financial risk"]:
            enhanced_text.append(line + ":")
            if (idx + 1 < len(lines)):
                enhanced_text.append("  " + lines[idx + 1])
            enhanced_text.append("\n")

        # Page numbers
        elif re.match(r"^\d{1,3} \d{1,3}$", line):
            continue

        else:
            enhanced_text.append(line)

    # Concatenate all the enhanced text into a single string
    cleaned_file = "\n".join(enhanced_text)

    return cleaned_file


def embed_index(doc_list):
    index_root = "./data_indexes/annual_reports" 
    print(f"Embedding {len(doc_list)} documents...")

    try:
        faiss_db = FAISS.from_documents(doc_list, embed_fn)  
    except Exception:
        faiss_db = FAISS.from_texts(doc_list, embed_fn)
    
    if os.path.exists(index_root):
        print("Merging with existing FAISS index...")
        local_db = FAISS.load_local(index_root, embed_fn)
        local_db.merge_from(faiss_db)
        local_db.save_local(index_root)
        print("Merge completed and index saved.")
    else:
        print("Creating a new FAISS index...")
        faiss_db.save_local(folder_path=index_root)
        print("New index saved.")

def data_process_indexes():
    patterns = ["*.txt", "*.pdf", "*.docx"]
    
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                file_path = os.path.join(root, name)
                text_content = extract_text_from_file(file_path)

                text_content = clean_pdf_text(text_content)
            
                add_documents(text_content)

        if all_documents:
            embed_index(all_documents)
        else:
            print("No documents found for embedding in this directory.")
        
        all_documents.clear()

data_process_indexes()
print("Script execution completed!")
