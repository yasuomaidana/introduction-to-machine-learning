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

# New prompt templates for the ReAct agent
REACT_PROMPT_TEMPLATES = {
    model_id: PromptTemplate.from_template(
        template.format(
            system_prompt='''
    {system_prompt}

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
    ```
    ''',
            user_prompt='''
    Begin!
    
    Question: {input}
    Thought:{agent_scratchpad}
    ''',
            format_prompt=""
        )
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
    return response


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
        max_iterations=3
    )

    # Invoke the agent
    return agent_executor.invoke({
        "input": user_prompt,
        "system_prompt": system_prompt,
        "tools": render_text_description(tools),
        "tool_names": ", ".join([t.name for t in tools])
    })


def calculator(expression: str) -> str:
    """A simple calculator for mathematical expressions.
    Input  MUST be a simple math expression like '25 + 63' or '15 * 7'."""
    try:
        # HINT: Use Python's eval() function for simple calculations
        return f"Result: {eval(expression.strip().strip("\"'`"))}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input MUST be in the exact format: 'format_type: text'
    Examples: 'uppercase: hello world', 'titlecase: langchain is awesome'"""
    try:
        # Parse the input to get format type and text
        # Your code here
        if len(formatted := text.split(":")) > 1:
            format_type, content = formatted[0].strip(), formatted[1].strip().lower()
            if format_type == "uppercase":
                return content.upper()
            elif format_type == "lowercase":
                return content.lower()
            elif format_type == "titlecase":
                return content.title()
            else:
                return f"Unknown format type: {format_type}"
        else:
            return f"Missing format. Example: titlecase: {text} -> {text.title()}"
    except Exception as e:
        return f"Error formatting text: {str(e)}"


# Create Tool objects for our functions
# HINT: Use the Tool class to wrap the functions
tools_list = [
    Tool(name="Calculator", func=calculator,
         description="A simple calculator that can add, subtract, multiply, or divide two numbers."),
    Tool(name="Text Formatter", func=format_text,
         description="Format text to uppercase, lowercase, or title case.")
]