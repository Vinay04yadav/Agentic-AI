import os
from dotenv import load_dotenv
from pathlib import Path
# pathlib is Python’s standard library for working with filesystem paths. It lets you build and inspect paths using objects and operators rather than manually joining strings.
from langchain_community.document_loaders import PyPDFLoader # for document loading
from langchain_text_splitters import RecursiveCharacterTextSplitter # for text splitting (chunking)

# for openai embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# for vector store
from langchain_qdrant import QdrantVectorStore


pdf_path = Path(__file__).parent / "NodeJSNotes.pdf"
# __file__ is the path to the running python file.
# Path(__file__) turns that string into a path object.
# .parent gets the folder containing the script.
# / "NodeJSNotes.pdf" appends the filename to that folder.


# Loading of data is done here:
loader = PyPDFLoader(
    file_path = pdf_path
)
docs = loader.load() # loads the pdf as langchain document object (one per page)

# print(docs[12]) #12 will access the 13th document or 13th page


# Now we need to do chunking:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = text_splitter.split_documents(documents=docs)


# Now the next step in embedding:
load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vector store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("Indexing of documents done!")