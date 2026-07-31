from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")

prompt = """
Only reply with the answer. What is the capital of Canada?
"""

response = model.invoke(prompt)
print(response.content)
