from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ API key not found!")

openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)


# This function will be running inside a queue.
def process_query(user_query: str):
    print(f"Searching chunks...")
    search_result = vector_db.similarity_search(query=user_query)


    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result])

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    context:
    {context}
    """

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
    print(f"Bot>>> {response.choices[0].message.content}")
    return response.choices[0].message.content