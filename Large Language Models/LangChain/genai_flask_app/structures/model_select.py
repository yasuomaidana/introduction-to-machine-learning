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
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")


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
            "Execution mode:\n"
            "- FORMATTED: Use whenever the user explicitly requests JSON output, structured format, sentiment analysis, ratings, or classification.\n"
            "- TOOL_USAGE: Use ONLY when the user needs math calculations (e.g. arithmetic, calculations) or specific text formatting tools.\n"
            "- SIMPLE: Use for standard conversational answers."
        )
    )
    schema_type: Schema | None = Field(
        default=None,
        description="Required if mode is FORMATTED: 'SENTIMENT' (for reviews, sentiment, classification) or 'GENERAL' (for other structured JSON outputs)."
    )
