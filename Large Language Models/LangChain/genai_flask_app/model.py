# Function to initialize a model
from langchain_core.prompts import PromptTemplate
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

# Prompt Templates

## ORIGINALS
# Prompt templates
# llama_template = PromptTemplate(
#     template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {system_prompt}\n{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
# {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# ''',
#     input_variables=["system_prompt", "format_prompt", "user_prompt"]
# )
#
# granite_template = PromptTemplate(
#     template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
#     input_variables=["system_prompt", "format_prompt", "user_prompt"]
# )
#
# mistral_template = PromptTemplate(
# 		template="<s>[INST]{system_prompt}\n{format_prompt}\n{user_prompt}[/INST]",
#     input_variables=["system_prompt", "format_prompt", "user_prompt"]
# )


# Documentation:
# Llama 3.1: https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama3_1/
# Hermes 3 Llama 3.1: https://ollama.com/finalend/hermes-3-llama-3.1:8b/blobs/b91960c3d374
# TinyLlama: https://ollama.com/library/tinyllama:latest

common_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''
llama_template = PromptTemplate(
    template=common_template,
    input_variables=["system_prompt", "user_prompt"]
)

# Based on the modelfile, Hermes 3 uses the ChatML prompt format.
hermes_chat_template = '''<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
'''
hermes_template = PromptTemplate(
    template=hermes_chat_template,
    input_variables=["system_prompt", "user_prompt"]
)

# Based on the ollama modelfile, TinyLlama uses a different format.
tinyllama_chat_template = '''<|system|>
{system_prompt}</s>
<|user|>
{user_prompt}</s>
<|assistant|>
'''
tinyllama_template = PromptTemplate(
    template=tinyllama_chat_template,
    input_variables=["system_prompt", "user_prompt"]
)
