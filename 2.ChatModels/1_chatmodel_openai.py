from lanchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.9)

result= model.invoke("What is the capital of India?")
print(result.content)

