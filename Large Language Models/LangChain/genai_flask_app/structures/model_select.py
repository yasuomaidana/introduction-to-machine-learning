from enum import Enum

from pydantic import BaseModel, Field


class ModelID(str, Enum):
    LLAMA_MODEL_ID = "llama3.2"
    TINYLLAMA_MODEL_ID = "tinyllama"
    HERMES_MODEL_ID = "hermes3:8b"


class Mode(str, Enum):
    TOOL_USAGE = "TOOL_USAGE"
    FORMATTED = "FORMATTED"
    SIMPLE = "SIMPLE"


class SentimentResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message up to two sentences")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")


class GeneralResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    response: str = Field(description="Suggested response to the user")


class Schema(str, Enum):
    SENTIMENT = "SENTIMENT"
    GENERAL = "GENERAL"


class ModelSelection(BaseModel):
    model_id: ModelID = Field(
        description="ID of the model to use: 'llama3.2', 'tinyllama', or 'hermes3:8b'."
    )
    mode: Mode = Field(
        description=(
            "Routing classification mode:\n"
            "- TOOL_USAGE: MUST be selected if the user request involves math equations, arithmetic, calculations (e.g. '6*4^3', '2+2', numbers), or explicit text formatting (uppercase, titlecase).\n"
            "- FORMATTED: MUST be selected ONLY if the user explicitly asks for JSON output, sentiment analysis, or structured classification.\n"
            "- SIMPLE: Default for standard questions, chat, or explanations."
        )
    )
    schema_type: Schema | None = Field(
        default=None,
        description="MUST be set to null if mode is TOOL_USAGE or SIMPLE. If mode is FORMATTED, set to 'SENTIMENT' (for reviews/classification) or 'GENERAL'."
    )
