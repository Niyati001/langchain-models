from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from scikit.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding= OpenAIEmbeddings(model= 'text-embedding-3-large', dimensions=300)

documents= [ 
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. Kohli has numerous records to his name, including being the fastest player to score 8,000, 9,000, 10,000, and 11,000 runs in One Day Internationals (ODIs). He has also been awarded the Sir Garfield Sobers Trophy for the ICC Cricketer of the Year multiple times. Kohli's aggressive playing style and consistent performances have made him a fan favorite and a prominent figure in international cricket.",
    "Sachin Tendulkar is a former Indian cricketer and one of the greatest batsmen in the history of cricket. He holds numerous records, including being the highest run-scorer in both Test and One Day International (ODI) cricket. Tendulkar was the first player to score a double century in ODIs and has scored 100 international centuries. He has received several awards, including the Bharat Ratna, India's highest civilian award. Tendulkar's skill, dedication, and sportsmanship have made him an iconic figure in the world of cricket.",
    "MS Dhoni is a former Indian cricketer and captain of the Indian national team in limited-overs formats. He is known for his calm demeanor, sharp cricketing acumen, and finishing abilities in matches. Dhoni led India to numerous victories, including the 2007 ICC World Twenty20, the 2011 ICC Cricket World Cup, and the 2013 ICC Champions Trophy. He is also one of the most successful captains in Indian cricket history. Dhoni's leadership qualities and contributions to Indian cricket have earned him immense respect and admiration from fans and fellow cricketers alike.",
    "Rohit Sharma is an Indian cricketer and the current captain of the Indian national team in limited-overs formats. He is known for his elegant batting style, ability to play long innings, and record-breaking performances in One Day Internationals (ODIs). Sharma holds the record for the highest individual score in ODIs, having scored 264 runs in a single match. He has also been a key player in India's victories in various international tournaments. Rohit Sharma's consistency, leadership skills, and contributions to Indian cricket have made him one of the most respected cricketers in the world.",
    "jasprit Bumrah is an Indian cricketer known for his exceptional fast bowling skills. He has been a crucial member of the Indian national team in all formats of the game. Bumrah is recognized for his unique bowling action, ability to bowl yorkers consistently, and his effectiveness in the death overs. He has played pivotal roles in India's victories in various international tournaments, including the ICC Cricket World Cup and ICC Champions Trophy. Bumrah's talent, dedication, and impact on Indian cricket have earned him widespread acclaim and respect among cricket enthusiasts."
]

query= "Who is the best Indian cricketer?"

doc_embeddings= embedding.embed_documents(documents)
query_embedding= embedding.embed_query(query)

scores= cosine_similarity([query_embedding], doc_embeddings)[0]
print(sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True))

print(query)
print(documents[index])
print("Similarity Score:", scores[index])