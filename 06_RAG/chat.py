from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os


# The user queries are supposed to be embedded from same model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# We need to point to our vector_db existing collection this time
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

# Take the user input
user_query = input(">>> ")

# Now we need to do similarity search on query and db
search_result = vector_db.similarity_search(query=user_query) # Relevant chunks are returned in search_results

# This variable collects the PDF passages returned by the similarity search, along with each passage’s page number and source path. Those details are added to the system prompt so the model can answer from the retrieved material and point us to the relevant page.
context_parts = []

for result in search_result:
    context_parts.append(
        f"Page Content: {result.page_content}\n"
        f"Page Number: {result.metadata['page_label']}\n"
        f"File Location: {result.metadata['source']}"
    )

context = "\n\n\n".join(context_parts)

SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

context:
{context}
"""
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("API Key not found")


openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

response = openai_client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role" : "system",
            "content" : SYSTEM_PROMPT
        },
        {
            "role" : "user",
            "content" : user_query
        }
    ]
)

print(f"Bot> {response.choices[0].message.content}")