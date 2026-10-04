from lanchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model_name="gemini-1.5-pro", temperature=0.9)
result= model.invoke("What is the capital of India?")
print(result.content)