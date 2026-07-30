from typing import TypedDict, List, Annotated
import operator

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_ollama import OllamaLLM
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Set up the language model
model_id = 'llama3.2'
parameters = {
    "num_predict": 256,
    "temperature": 0.2,
}

# Initialize the model
llm = OllamaLLM(
    model=model_id,
    **parameters
)

# --- Part 1: ConversationBufferMemory Simulation using LangGraph ---

# 2. Define the state and graph for buffer memory
class BufferAgentState(TypedDict):
    messages: Annotated[list, operator.add]

buffer_graph_builder = StateGraph(BufferAgentState)

def chatbot_node(state: BufferAgentState):
    """Invokes the LLM and wraps the response in an AIMessage."""
    response_str = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=response_str)]}

buffer_graph_builder.add_node("chatbot", chatbot_node)
buffer_graph_builder.set_entry_point("chatbot")
buffer_graph_builder.set_finish_point("chatbot")

buffer_memory = MemorySaver()
buffer_memory_graph = buffer_graph_builder.compile(checkpointer=buffer_memory)

# 3. Run simulation for buffer memory
test_inputs = [
    "My favorite color is blue.",
    "I enjoy hiking in the mountains.",
    "What activities would you recommend for me?",
    "What was my favorite color again?",
    "Can you remember both my name and my favorite color?"
]

print("\n=== Beginning Chat Simulation (Buffer Memory) ===")
thread = {"configurable": {"thread_id": "buffer_thread"}}
initial_messages = [
    HumanMessage(content="Hello, my name is Alice."),
    AIMessage(content="Hello Alice! It's nice to meet you. How can I help you today?")
]
buffer_memory_graph.update_state(thread, {"messages": initial_messages})

for i, user_input in enumerate(test_inputs):
    print(f"\n--- Turn {i+1} ---")
    print(f"Human: {user_input}")
    response = buffer_memory_graph.invoke({"messages": [HumanMessage(content=user_input)]}, thread)
    ai_message = response["messages"][-1]
    print(f"AI: {ai_message.content}")

final_buffer_state = buffer_memory_graph.get_state(thread)
final_buffer_memory = "".join([msg.content for msg in final_buffer_state.values['messages']])
print("\n\nFinal Buffer Memory Contents (Full History):")
print(final_buffer_memory)


# --- Part 2: ConversationSummaryMemory Simulation using LangGraph ---
print("\n\n===== PART 2: Conversation Summary Memory")
# 4. Define the state and graph for summary memory
class SummaryAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    summary: str

def summary_chatbot_node(state: SummaryAgentState):
    """Invokes the LLM with a summary and wraps the response in an AIMessage."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Here is a summary of the conversation so far:\n{summary}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | llm
    response_str = chain.invoke(state)
    return {"messages": [AIMessage(content=response_str)]}

def summarizer_node(state: SummaryAgentState):
    """Summarizes the conversation history."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation. Include key details like names and preferences. If the conversation is short, keep the summary concise."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | llm
    summary = chain.invoke({"messages": state["messages"]})
    return {"summary": summary}

summary_graph_builder = StateGraph(SummaryAgentState)
summary_graph_builder.add_node("chatbot", summary_chatbot_node)
summary_graph_builder.add_node("summarizer", summarizer_node)
summary_graph_builder.set_entry_point("chatbot")
summary_graph_builder.add_edge("chatbot", "summarizer")
summary_graph_builder.add_edge("summarizer", END)

summary_memory = MemorySaver()
summary_memory_graph = summary_graph_builder.compile(checkpointer=summary_memory)

# 5. Run simulation for summary memory
print("\n\n=== Testing Conversation Summary Memory ===")
summary_thread = {"configurable": {"thread_id": "summary_thread"}}

# Initialize the state with a starting summary
summary_memory_graph.update_state(summary_thread, {
    "messages": [
        HumanMessage(content="Hello, my name is Alice."),
        AIMessage(content="Hello Alice! It's nice to meet you. How can I help you today?")
    ],
    "summary": "The user's name is Alice."
})

for i, user_input in enumerate(test_inputs):
    print(f"\n--- Turn {i+1} ---")
    print(f"Human: {user_input}")
    # We only need to pass the new message. The graph will load the rest from the checkpointer.
    response = summary_memory_graph.invoke({"messages": [HumanMessage(content=user_input)]}, summary_thread)
    ai_message = response["messages"][-1]
    print(f"AI: {ai_message.content}")

final_summary_state = summary_memory_graph.get_state(summary_thread)
final_summary = final_summary_state.values['summary']
print("\n\nFinal Summary Memory Contents:")
print(final_summary)


# --- Part 3: Comparison ---
print("\n\n=== Memory Comparison ===")
print(f"Buffer Memory Size: {len(final_buffer_memory)} characters")
print(f"Summary Memory Size: {len(final_summary)} characters")
print("\nThe conversation summary memory creates a more compact representation of the chat history.")
