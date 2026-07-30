# LangChain Modern API Cheat Sheet (v0.2 / v0.3+)

A comprehensive, updated reference guide for building LLM applications with modern **LangChain**, **LangChain Expression Language (LCEL)**, and local Ollama integrations.

---

## 📌 Summary & Quick Reference Table

| Category             | Component / Class                            | Primary Import Path                          | Purpose / Description                                          |
|----------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------------------------|
| **Messages**         | `HumanMessage`, `AIMessage`, `SystemMessage` | `langchain_core.messages`                    | Represents user, AI assistant, and system instructions         |
| **Prompts**          | `PromptTemplate`, `ChatPromptTemplate`       | `langchain_core.prompts`                     | Formats dynamic text and chat message templates                |
| **Prompts**          | `MessagesPlaceholder`                        | `langchain_core.prompts`                     | Slots dynamic conversation history into prompt templates       |
| **Output Parsers**   | `StrOutputParser`                            | `langchain_core.output_parsers`              | Extracts string text from `AIMessage` output                   |
| **Output Parsers**   | `JsonOutputParser`                           | `langchain_core.output_parsers`              | Parses LLM output into JSON conforming to Pydantic v2 schemas  |
| **Output Parsers**   | `CommaSeparatedListOutputParser`             | `langchain_core.output_parsers`              | Converts comma-delimited response into a Python list           |
| **Models**           | `ChatOllama`, `OllamaLLM`                    | `langchain_ollama`                           | Local LLM and ChatModel interfaces via Ollama                  |
| **Document Loaders** | `PyPDFLoader`, `WebBaseLoader`               | `langchain_community.document_loaders`       | Loads PDF files or Web HTML into `Document` objects            |
| **Text Splitters**   | `RecursiveCharacterTextSplitter`             | `langchain_text_splitters`                   | Splits documents recursively based on natural text boundaries  |
| **Text Splitters**   | `CharacterTextSplitter`                      | `langchain_text_splitters`                   | Splits document text on specified character separators         |
| **Embeddings**       | `OllamaEmbeddings`                           | `langchain_ollama`                           | Generates vector embeddings for text using local Ollama        |
| **Vector Stores**    | `Chroma`                                     | `langchain_chroma`                           | Stores vector embeddings and enables similarity searches       |
| **Retrievers**       | `VectorStoreRetriever`                       | `vectorstore.as_retriever()`                 | Wraps vector store to retrieve documents via `.invoke()`       |
| **Retrievers**       | `ParentDocumentRetriever`                    | `langchain.retrievers`                       | Embeds small child chunks but retrieves full parent context    |
| **RAG Chains**       | `create_retrieval_chain`                     | `langchain.chains`                           | Connects a retriever & doc chain into a RAG pipeline           |
| **LCEL Piping**      | `RunnablePassthrough`                        | `langchain_core.runnables`                   | Enables step-by-step pipeline composition (`.assign()`)        |
| **Memory**           | `ChatMessageHistory`                         | `langchain_community.chat_message_histories` | In-memory container for managing conversation history          |
| **Memory**           | `RunnableWithMessageHistory`                 | `langchain_core.runnables.history`           | Wraps LCEL chains for automatic session-based state management |
| **Tools**            | `@tool` decorator, `Tool`                    | `langchain_core.tools`                       | Defines callable tools for agents to execute actions           |
| **Agents**           | `create_react_agent`, `AgentExecutor`        | `langchain.agents`                           | Constructs and runs ReAct reasoning loops with tools           |

---

## 1. Message Types

Represent standard roles in conversational models.

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Define a list of messages
messages = [
    SystemMessage(content="You are a helpful AI assistant that recommends books in one short sentence."),
    HumanMessage(content="I enjoy mystery novels, what should I read?")
]

# Invoke model with message list
response = chat_model.invoke(messages)
# Returns: AIMessage(content="I recommend 'Gone Girl' by Gillian Flynn for its suspenseful plot.")
```

---

## 2. Prompt Templates

Construct dynamic prompts with variables.

### String PromptTemplate
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}.")

# Invoke template to get a formatted PromptValue
formatted_prompt = prompt.invoke({"adjective": "funny", "topic": "cats"})
# Output string: prompt.format(adjective="funny", topic="cats")
```

### ChatPromptTemplate
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("user", "Tell me a joke about {topic}")
])

formatted_messages = prompt.invoke({"domain": "science", "topic": "cats"})
```

### MessagesPlaceholder
Slot a dynamic list of past messages into a specific spot in a chat prompt.

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("user", "{input}")
])

formatted = prompt.invoke({
    "history": [HumanMessage(content="What is the day after Tuesday?")],
    "input": "And what month follows June?"
})
```

---

## 3. Output Parsers & Structured Outputs

Transform raw text/AIMessage responses into structured data.

### StrOutputParser
Extracts the string content directly from an `AIMessage`.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
# Used in LCEL: chain = prompt | model | StrOutputParser()
```

### JsonOutputParser (with Pydantic v2)
Uses standard Pydantic models to format and parse JSON outputs.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="Question setting up the joke")
    punchline: str = Field(description="Answer resolving the joke")

parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\nQuery: {query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
result = chain.invoke({"query": "Tell me a joke."})
# Returns Python dict: {"setup": "...", "punchline": "..."}
```

### CommaSeparatedListOutputParser
Parses comma-separated output into a Python list.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
result = chain.invoke({"subject": "ice cream flavors"})
# Returns: ['Vanilla', 'Chocolate', 'Strawberry', 'Mint Chocolate Chip', 'Cookie Dough']
```

### `.with_structured_output()` (Native Model Structured Output)
Modern method supported by tool-calling ChatModels (e.g. OpenAI, Ollama, Anthropic).

```python
structured_model = model.with_structured_output(Joke)
joke_obj = structured_model.invoke("Tell me a joke about computers.")
```

---

## 4. Models & Providers (Ollama & OpenAI)

Using LLMs and Chat Models (e.g. local Ollama or OpenAI).

```python
# Local Ollama (langchain-ollama package)
from langchain_ollama import ChatOllama, OllamaLLM

# Text model (LLM interface)
llm = OllamaLLM(model="llama3.2", temperature=0.2)

# Chat model interface
chat_model = ChatOllama(model="llama3.2", temperature=0.2)

# OpenAI (langchain-openai package)
# from langchain_openai import ChatOpenAI
# chat_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
```

---

## 5. Document Loaders & Text Splitters

### Document Object
Container for text and metadata.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Python is a high-level programming language.",
    metadata={"source": "docs", "page": 1}
)
```

### Document Loaders
```python
# Web Pages (langchain-community)
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
docs = loader.load()

# PDF (PyPDF)
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Modern PDF Loaders (Docling / Unstructured for complex layouts & tables)
# from langchain_docling.loader import DoclingLoader
# loader = DoclingLoader(file_path="document.pdf")
# docs = loader.load()
```

### Text Splitters (`langchain-text-splitters`)
> **Note**: Splitters are now imported from `langchain_text_splitters` (formerly `langchain.text_splitter`).

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitter (Recommended for general text)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(docs)

# CharacterTextSplitter (Simple single-separator split)
char_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
chunks = char_splitter.split_documents(docs)
```

---

## 6. Vector Stores & Embeddings

### Embeddings (`langchain-ollama`)
```python
from langchain_ollama import OllamaEmbeddings

# Local Ollama Embeddings (nomic-embed-text is recommended)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Vector Store (`langchain-chroma`)
> **Note**: Use `langchain_chroma` instead of deprecated `langchain.vectorstores.Chroma`.

```python
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Create vector store from documents (filtering unsupported metadata types)
vectorstore = Chroma.from_documents(
    documents=filter_complex_metadata(chunks),
    embedding=embeddings,
    collection_name="my_collection"
)

# Similarity Search
results = vectorstore.similarity_search("What is LangChain?", k=3)
```

---

## 7. Retrievers & RAG Pipelines

### Vector Store Retriever
Retrievers standardly use `.invoke()` instead of deprecated `.get_relevant_documents()`.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Retrieve relevant documents
retrieved_docs = retriever.invoke("What is LangChain?")
```

### Modern RAG Chain (`create_retrieval_chain`)
The updated standard for Retrieval-Augmented Generation.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Create document combination prompt & chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the provided context:\n\n{context}"),
    ("user", "{input}")
])
combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)

# 2. Combine with retriever
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 3. Invoke RAG chain
response = rag_chain.invoke({"input": "What is LangChain?"})
# Output dictionary contains: response["answer"] and response["context"]
```

### ParentDocumentRetriever
Retrieves full parent document context while searching on smaller child chunk embeddings.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

vectorstore = Chroma(collection_name="parent_child", embedding_function=embeddings)
store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add raw documents (retriever handles splitting and embedding internally)
parent_retriever.add_documents(docs)

# Invoke retriever
full_context_docs = parent_retriever.invoke("LangChain concepts")
```

---

## 8. LangChain Expression Language (LCEL)

LCEL is the standard pattern for composing chains, replacing legacy `LLMChain` and `SequentialChain`.

### Simple LCEL Chain
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Recommend a classic dish from {location}.")
chain = prompt | chat_model | StrOutputParser()

dish = chain.invoke({"location": "Mexico"})
```

### Multi-Step Sequential Chain (`RunnablePassthrough.assign`)
Pass outputs along a multi-step sequence without losing prior inputs.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Step 1: Suggest dish
dish_chain = PromptTemplate.from_template("Suggest a classic dish from {location}.") | chat_model | StrOutputParser()

# Step 2: Recipe for dish
recipe_chain = PromptTemplate.from_template("Give a simple recipe for {meal}.") | chat_model | StrOutputParser()

# Step 3: Cooking time for recipe
time_chain = PromptTemplate.from_template("Estimate cooking time for this recipe: {recipe}.") | chat_model | StrOutputParser()

# Combine with RunnablePassthrough.assign
overall_chain = (
    RunnablePassthrough.assign(meal=lambda x: dish_chain.invoke({"location": x["location"]}))
    | RunnablePassthrough.assign(recipe=lambda x: recipe_chain.invoke({"meal": x["meal"]}))
    | RunnablePassthrough.assign(time=lambda x: time_chain.invoke({"recipe": x["recipe"]}))
)

result = overall_chain.invoke({"location": "Japan"})
# Returns dict with keys: 'location', 'meal', 'recipe', 'time'
```

---

## 9. Memory & Conversation History

### ChatMessageHistory
Lightweight object to store and manage message history manually.

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Hello, my name is Alice.")
history.add_ai_message("Hi Alice! How can I assist you today?")

# Pass history messages directly to chat model
response = chat_model.invoke(history.messages)
history.add_ai_message(response.content)
```

### `RunnableWithMessageHistory` (Modern LCEL State Management)
Wraps any LCEL chain to inject and automatically manage conversation history per session.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Message history store per session
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | chat_model

with_history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Invoke with session_id in config
res1 = with_history_chain.invoke(
    {"input": "My favorite color is blue."},
    config={"configurable": {"session_id": "user_123"}}
)

res2 = with_history_chain.invoke(
    {"input": "What is my favorite color?"},
    config={"configurable": {"session_id": "user_123"}}
)
```

---

## 10. Tools & ReAct Agents

### Tool Definition

#### Option A: `@tool` Decorator (Recommended)
```python
from langchain_core.tools import tool

@tool
def calculate_expression(expression: str) -> str:
    """Useful for evaluating simple mathematical expressions.
    Input must be a valid python math string e.g. '25 + 63'.
    """
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"
```

#### Option B: `Tool` Class
```python
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
python_tool = Tool(
    name="Python_Calculator",
    func=python_repl.run,
    description="Executes Python code. Useful for mathematical calculations."
)
```

### ReAct Agent Execution (`create_react_agent` & `AgentExecutor`)
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

tools = [calculate_expression]

# Standard ReAct prompt format
react_template = """You are an agent with access to tools:

{tools}

Tool names: {tool_names}

To use a tool:
Thought: I need to figure out what to do
Action: tool_name
Action Input: input_to_tool
Observation: tool_result

Repeat Thought-Action-Observation cycle until final answer.
Final output format:
Thought: I know the final answer
Final Answer: your final response

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

# Create agent and executor
agent = create_react_agent(llm=chat_model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Run agent
result = agent_executor.invoke({"input": "What is the square root of 256 plus 42?"})
print(result["output"])
```

---

## 11. Deprecation & Migration Guide

| Old / Deprecated Import or API | Modern / Updated Replacement | Package |
|---|---|---|
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` | `langchain-text-splitters` |
| `from langchain.vectorstores import Chroma` | `from langchain_chroma import Chroma` | `langchain-chroma` |
| `from langchain.tools import tool` | `from langchain_core.tools import tool` | `langchain-core` |
| `from langchain_community.chat_models import ChatOllama` | `from langchain_ollama import ChatOllama` | `langchain-ollama` |
| `from langchain_community.embeddings import OllamaEmbeddings` | `from langchain_ollama import OllamaEmbeddings` | `langchain-ollama` |
| `from langchain_core.pydantic_v1 import ...` | `from pydantic import ...` | `pydantic` (v2) |
| `LLMChain(...)` | `prompt \| model \| output_parser` | LCEL (`langchain-core`) |
| `SequentialChain(...)` | `RunnablePassthrough.assign(...)` | LCEL (`langchain-core`) |
| `retriever.get_relevant_documents(query)` | `retriever.invoke(query)` | `langchain-core` |
| `RetrievalQA.from_chain_type(...)` | `create_retrieval_chain(retriever, combine_docs_chain)` | `langchain` |
