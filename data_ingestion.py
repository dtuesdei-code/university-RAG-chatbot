from importlib import metadata
from pydoc import doc

from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
import re
import os

from sympy import content

loader =DirectoryLoader(
    "./data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader

)

my_documents=loader.load()
full_text="\n".join(doc.page_content for doc in my_documents)
#print(full_text)


chunked_program=re.split("Degree Programme",full_text,flags=0)
structured_doc=[]
for prog in chunked_program:
    program_match=re.search(r"^(.*?)\n",prog)
    program_name=program_match.group(1).strip() if program_match else "unknown program"

    years=re.split(r"Year (one|two|three|four)",prog, flags=re.IGNORECASE)
    for i in range(1,len(years),2):
        year=years[i].strip()
        year_content=years[i+1].strip()
        
        semester=re.split(r"Semester (one|two)",year_content,flags=re.IGNORECASE)
        for j in range(1,len(semester),2):
            semester_name=semester[j].strip()
            semester_content=semester[j+1].strip()

            structured_doc.append({
                "text":semester_content,
                "metadata":{
                    "program":program_name,
                    "year":year,
                    "semester":semester_name
                }
            })
            



def cleaned_courses(text):
    lines=text.split("\n")
    courses=[]

    for line in lines:
        line=line.strip()

        if not line:
            continue
        if "course" in line:
            continue

        courses.append(line)


    return "courses\n "+ "\n".join(f"-{c}" for c in courses)

document_to_embedd=[]
for doc in structured_doc:
    cleaned_text=cleaned_courses(doc["text"])
    document_to_embedd.append(Document(
        page_content=cleaned_text,
        metadata=doc["metadata"]
    ))



            





