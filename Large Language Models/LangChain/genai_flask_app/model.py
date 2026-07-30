# Function to initialize a model
from typing import Type, TypeVar

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

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

MODELS = {
    LLAMA_MODEL_ID: llama_llm,
    TINYLLAMA_MODEL_ID: tinyllama_llm,
    HERMES_MODEL_ID: hermes_llm
}

# Documentation:
# Llama 3.1: https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama3_1/
# Hermes 3 Llama 3.1: https://ollama.com/finalend/hermes-3-llama-3.1:8b/blobs/b91960c3d374
# TinyLlama: https://ollama.com/library/tinyllama:latest

llama_chat_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

# Based on the modelfile, Hermes 3 uses the ChatML prompt format.
hermes_chat_template = '''<|im_start|>system
{system_prompt}
{format_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
'''

# Based on the ollama modelfile, TinyLlama uses a different format.
tinyllama_chat_template = '''<|system|>
{system_prompt}
{format_prompt}</s>
<|user|>
{user_prompt}</s>
<|assistant|>
'''

TEMPLATES = {
    LLAMA_MODEL_ID: llama_chat_template,
    TINYLLAMA_MODEL_ID: tinyllama_chat_template,
    HERMES_MODEL_ID: hermes_chat_template
}

PROMPT_TEMPLATES = {
    model_id: PromptTemplate(
        template=template,
        input_variables=["system_prompt", "user_prompt", "format_prompt"],
    )
    for model_id, template in TEMPLATES.items()
}

PROMPT_TEMPLATES_NO_PARSER = {
    model_id: PromptTemplate(
        template=template.replace("{format_prompt}", ""),
        input_variables=["system_prompt", "user_prompt"],
    )
    for model_id, template in TEMPLATES.items()
}


def get_ai_response(model_id: str, system_prompt: str, user_prompt: str):
    if model_id not in PROMPT_TEMPLATES_NO_PARSER or model_id not in MODELS:
        raise ValueError(f"Invalid model ID: {model_id}")
    model = MODELS[model_id]
    template = PROMPT_TEMPLATES_NO_PARSER[model_id]
    chain = template | model
    response = chain.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
    return response.content


T = TypeVar('T', bound=BaseModel)


def get_ai_json_response(model_id: str, system_prompt: str, user_prompt: str, json_type: Type[T]) -> T:
    if model_id not in PROMPT_TEMPLATES or model_id not in MODELS:
        raise ValueError(f"Invalid model ID: {model_id}")
    if not issubclass(json_type, BaseModel):
        raise ValueError(f"json_type must be a subclass of BaseModel, got {json_type}")

    model = MODELS[model_id]
    output_parser = JsonOutputParser(pydantic_object=json_type)
    format_prompt = output_parser.get_format_instructions()

    template = PROMPT_TEMPLATES[model_id]
    chain = template | model | output_parser

    response = chain.invoke({
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "format_prompt": format_prompt
    })
    return response
