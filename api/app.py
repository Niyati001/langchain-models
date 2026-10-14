
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langserve import add_routes
import uvicorn

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A server for LangChain applications using Ollama"
)

# Local Llama 3.2 model through Ollama
model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

# Prompts
prompt1 = ChatPromptTemplate.from_template(
    "Write an essay about {topic} in approximately 100 words."
)

prompt2 = ChatPromptTemplate.from_template(
    "Write a poem about {topic} in approximately 100 words."
)

# API endpoints
add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | model,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
