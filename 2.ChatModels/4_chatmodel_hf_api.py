from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
repo_id="Qwen/Qwen2.5-7B-Instruct-1M",    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm, temperature=0.9)

result = model.invoke("What is the capital of India?")
print(result.content)