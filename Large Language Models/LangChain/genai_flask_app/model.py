# Function to initialize a model
from langchain_ollama import ChatOllama

from genai_flask_app.config import PARAMETERS, LLAMA_MODEL_ID, TINYLLAMA_MODEL_ID, HERMES_MODEL_ID


def initialize_model(model_id) -> ChatOllama:
    if model_id not in PARAMETERS:
        raise ValueError(f"Invalid model ID: {model_id}")
    return ChatOllama(
        **PARAMETERS[model_id]
    )


# Initialize models
llama_llm = initialize_model(LLAMA_MODEL_ID)
tinyllama_llm = initialize_model(TINYLLAMA_MODEL_ID)
hermes_llm = initialize_model(HERMES_MODEL_ID)


# https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama3_1/

# https://ollama.com/library/tinyllama:latest

# https://ollama.com/finalend/hermes-3-llama-3.1:8b/blobs/b91960c3d374