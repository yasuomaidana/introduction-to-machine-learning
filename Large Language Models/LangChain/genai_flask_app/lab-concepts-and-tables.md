# Extracted Conceptual Content and Tables

## 1. Project Purpose

The lab focuses on building a GenAI Flask application with structured AI output, modular design, and reusable
model-integration patterns.

Two delivery variants are covered:

- **IBM cloud stack:** `ibm-watsonx-ai` + `langchain-ibm` + Watsonx models (Llama, Granite, Mistral variants).
- **Local stack:** `langchain-community` + Ollama local models (Llama, Mistral).

---

## 2. Learning Objectives (Concepts)

- Build Flask apps with AI endpoints.
- Integrate multiple LLMs behind a consistent interface.
- Use prompt engineering for reliable responses.
- Produce **structured JSON outputs** via LangChain output parsers.
- Compare models across capability, quality, cost, and speed.
- Keep code modular and maintainable for future model swaps.

---

## 3. Core LLM Concepts

### How LLMs Work

LLMs learn statistical language patterns from very large corpora and generate likely next tokens in context.

### Tokenization

Text is split into tokens (subwords/word pieces), mapped to IDs, and processed numerically.

### Embeddings

Token IDs are projected into vectors where semantic similarity is represented by geometric proximity.

### Attention

Attention lets each token condition on other tokens in the sequence to capture meaning and relationships.

### Transformer Layers

Stacked layers progressively build richer representations (syntax -> semantics -> higher-order behavior).

### Next-Token Prediction

Training objective: predict the next token; this drives language fluency and task behavior.

### Massive-Scale Training

Models are trained with huge datasets and many parameters using distributed optimization.

### Inference

Generation is autoregressive (token-by-token), with decoding controls (e.g., greedy/sampling) affecting determinism vs
creativity.

---

## 4. Model Selection Concepts

Model choice should be based on task-fit testing, not specs alone.

Key factors:

- **Capabilities** (text-only vs multimodal, coding, reasoning, etc.)
- **Cost** (input/output token pricing)
- **Speed** (latency/throughput)
- **Quality** (task accuracy and relevance)
- **Operational constraints** (vendor, licensing, integration, governance)

---

## 5. Prompting and Role/Token Formatting

Different model families require different prompt formats and control tokens. Correct formatting improves response
reliability and instruction following.

### Roles (Llama-context conceptual model)

- **System:** behavior and constraints for the assistant.
- **User:** incoming query/request.
- **Assistant:** model-generated reply.
- **iPython/tool-output role:** tool-return channel in certain tool-calling flows.

---

## 6. Structured Output Concepts

Raw text is often insufficient for production workflows; structured outputs are easier to validate, store, and automate.

Pattern used in both labs:

1. Define a response schema (Pydantic model).
2. Use `JsonOutputParser` to enforce schema-shaped output.
3. Inject parser formatting instructions into prompts.
4. Chain prompt -> model -> parser for deterministic downstream handling.

Common schema ideas used in lab examples include: summary, sentiment, response, and optional action/category fields.

---

## 7. LangChain Concepts

LangChain provides:

- A consistent abstraction over different model providers.
- Prompt templating and composable chains.
- Output parsers for structured JSON contracts.
- Cleaner support for multi-step AI workflows.

Benefits emphasized:

- Faster model swapping with minimal app rewiring.
- Reusable modular components.
- More maintainable, production-oriented GenAI code.

---

## 8. Architecture and App-Design Concepts

- Separate configuration from model logic.
- Create model-specific wrappers around shared chain logic.
- Keep Flask routes thin; delegate AI behavior to model modules.
- Include input validation, error responses, and response timing.
- Maintain frontend/backend separation (templates + static assets).

---

## 9. Environment Concepts

### IBM variant

- Cloud-hosted Watsonx endpoints.
- Configurable generation parameters (e.g., decoding method, max new tokens).
- Enterprise model catalog and pricing/performance tradeoffs.

### Local variant

- On-device/local-network inference via Ollama.
- No remote API dependency for core generation.
- Practical model comparison on local runtime constraints.

---

## 10. Key Takeaways and Forward Concepts

- Structured output and prompt discipline are central to reliable GenAI apps.
- Model behavior varies significantly; empirical evaluation is required.
- Modularity enables maintainability and future experimentation.
- Next maturity steps include caching, richer memory/context handling, A/B testing, stronger observability, and broader
  model coverage.

---

## 11. Extracted Tables

### 11.1 Watsonx Model Comparison Table

| Provider | Model ID                                          | Use Cases                                                                                                                                                      | Context Length | Price USD per million tokens |
|----------|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|------------------------------|
| IBM      | ibm/granite-4-h-small                             | Q&A, summarization, classification, generation, extraction, RAG, coding, and multi-tool agentic workflows.                                                     | 128k           | Input: 0.06 / Output: 0.25   |
| Meta     | meta-llama/llama-4-maverick-17b-128e-instruct-fp8 | Multimodal reasoning, long-context processing, code generation and analysis, multilingual operations (200 languages supported), STEM and logical reasoning.    | 1M             | Input: 0.35 / Output: 1.40   |
| Mistral  | mistralai/mistral-small-3-1-24b-instruct-2503     | Instruction following, conversational assistance, image understanding, function calling, multilingual Q&A, summarization, classification, generation, and RAG. | 128k           | Input: 0.10 / Output: 0.30   |
| Mistral  | mistralai/mistral-medium-2505                     | Programming, mathematical reasoning, long document understanding, summarization, dialogue, and multimodal (text + image) tasks.                                | 128k           | Input: 0.40 / Output: 2.00   |

### 11.2 Llama Special Tokens

| Token Name            | Description                                                                                                                                                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <\|begin_of_text\|>   | Specifies the start of the prompt.                                                                                                                                                                                              |
| <\|end_of_text\|>     | Specifies the end of the prompt.                                                                                                                                                                                                |
| <\|start_header_id\|> | These tokens enclose the role for a particular message, always paired with <\|end_header_id\|>. The possible roles are: [system, user, assistant, and ipython]                                                                  |
| <\|end_header_id\|>   | Pairs with <\|start_header_id\|> to define role for a particular message                                                                                                                                                        |
| <\|eot_id\|>          | End of turn. Represents when the model has determined that it has finished interacting with the user message that initiated its response. This token signals to the executor that the model has finished generating a response. |

### 11.3 Mistral Tokens

| Token Name | Description                                                                                  |
|------------|----------------------------------------------------------------------------------------------|
|            | Marks the start of a sentence or sequence.                                                   |
| <\s>       | Marks the end of a sentence or sequence.                                                     |
| [INST]     | Signifies the start of an instructional message or command. Typically used for instructions. |
| [/INST]    | Marks the end of the instructional message.                                                  |

### 11.4 Granite Tokens

| Token Name      | Description                                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------|
| <\|system\|>    | Identifies the instruction, commonly referred to as the system prompt for the foundation model. |
| <\|user\|>      | The query text to be answered.                                                                  |
| <\|assistant\|> | A cue at the end of the prompt that indicates that a generated answer is expected.              |

