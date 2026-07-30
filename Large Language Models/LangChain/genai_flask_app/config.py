# Model IDs
LLAMA_MODEL_ID = "llama3.2"
TINYLLAMA_MODEL_ID = "tinyllama"
HERMES_MODEL_ID = "hermes3:8b"

# Llama 2 13b Model parameters
LLAMA_PARAMETERS = {
    "name": LLAMA_MODEL_ID,
    "num_predict": 200,
    "temperature": 0.5,
    "top_k": 6,
    "top_p": .75,
}

# TinyLlama Model parameters
TINYLLAMA_PARAMETERS = {
    "name": TINYLLAMA_MODEL_ID,
    "num_predict": 200,
    "temperature": 0.5,
    "top_k": 3,
    "top_p": 0.25,
}

HERMES_MODEL = {
    "name": HERMES_MODEL_ID,
    "reasoning": True,
    "num_predict": 200,
    "temperature": 0.5,
    "top_k": 5,
    "top_p": 0.3,
}

PARAMETERS = {
    LLAMA_MODEL_ID: LLAMA_PARAMETERS,
    TINYLLAMA_MODEL_ID: TINYLLAMA_PARAMETERS,
    HERMES_MODEL_ID: HERMES_MODEL
}
