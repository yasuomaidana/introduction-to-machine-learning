# Function to initialize a model
from typing import Type, TypeVar

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description
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

# ReAct prompt templates with correct role tag placement
REACT_SYSTEM_INSTRUCTION = """{system_prompt}

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```"""

REACT_LLAMA_TEMPLATE = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{REACT_SYSTEM_INSTRUCTION}<|eot_id|><|start_header_id|>user<|end_header_id|>
Begin!

Question: {{input}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Thought: {{agent_scratchpad}}"""

REACT_HERMES_TEMPLATE = f"""<|im_start|>system
{REACT_SYSTEM_INSTRUCTION}<|im_end|>
<|im_start|>user
Begin!

Question: {{input}}<|im_end|>
<|im_start|>assistant
Thought: {{agent_scratchpad}}"""

REACT_TINYLLAMA_TEMPLATE = f"""<|system|>
{REACT_SYSTEM_INSTRUCTION}</s>
<|user|>
Begin!

Question: {{input}}</s>
<|assistant|>
Thought: {{agent_scratchpad}}"""

REACT_PROMPT_TEMPLATES = {
    LLAMA_MODEL_ID: PromptTemplate.from_template(REACT_LLAMA_TEMPLATE),
    HERMES_MODEL_ID: PromptTemplate.from_template(REACT_HERMES_TEMPLATE),
    TINYLLAMA_MODEL_ID: PromptTemplate.from_template(REACT_TINYLLAMA_TEMPLATE),
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

    # Add the JSON instruction to the system prompt
    system_prompt_with_json_instruction = (
        f"{system_prompt}\nYou MUST reply with a valid JSON object with NO additional text, just the JSON."
    )

    template = PROMPT_TEMPLATES[model_id]
    chain = template | model | output_parser

    response = chain.invoke({
        "system_prompt": system_prompt_with_json_instruction,
        "user_prompt": user_prompt,
        "format_prompt": format_prompt
    })
    if isinstance(response, json_type):
        return response

    if isinstance(response, dict):
        return json_type.model_validate(response)
    raise TypeError(f"Expected dict or {json_type.__name__}, got {type(response).__name__}")


def get_ai_tool_response(model_id: str, system_prompt: str, user_prompt: str, tools: list[Tool]):
    if model_id not in REACT_PROMPT_TEMPLATES or model_id not in MODELS:
        raise ValueError(f"Invalid model ID: {model_id}")

    prompt = REACT_PROMPT_TEMPLATES[model_id]
    model = MODELS[model_id]

    # Create the ReAct agent
    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors="Check your format! Ensure you use Action: <tool_name> and Action Input: <input>",
        max_iterations=5,
        early_stopping_method="generate"
    )

    # Invoke the agent
    return agent_executor.invoke({
        "input": user_prompt,
        "system_prompt": system_prompt,
        "tools": render_text_description(tools),
        "tool_names": ", ".join([t.name for t in tools])
    })
