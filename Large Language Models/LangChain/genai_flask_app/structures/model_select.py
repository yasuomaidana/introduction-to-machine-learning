from enum import Enum

from pydantic import BaseModel


class ModelID(str, Enum):
    LLAMA_MODEL_ID = "llama3.2"
    TINYLLAMA_MODEL_ID = "tinyllama"
    HERMES_MODEL_ID = "hermes3:8b"


class Mode(str, Enum):
    TOOL_USAGE = "TOOL_USAGE"
    FORMATTED = "FORMATTED"
    SIMPLE = "SIMPLE"


class ModelSelection(BaseModel):
    model_id: ModelID

    mode: Mode
