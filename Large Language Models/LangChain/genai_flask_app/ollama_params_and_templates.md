# Hermes and Ollama (Llama 3.1) Parameters and Templates

This document outlines the specific parameters and templates for using the Hermes and Llama 3.1 models with Ollama.

## Hermes

### Parameters

The `stop` parameter is an array of strings that tells the model when to stop generating text. These are special tokens
that the model recognizes as indicators of the end of a turn or the beginning of a new one.

| Token                     | Description                              |
|:--------------------------|:-----------------------------------------|
| `<\| start_header_id \|>` | Marks the beginning of a message header. |
| `<\| end_header_id \|>`   | Marks the end of a message header.       |
| `<\| eot_id \|>`          | End of text token.                       |
| `<\| im_start \|>`        | Start of a message.                      |
| `<\| im_end \|>`          | End of a message.                        |
| `<\| begin_of_text \|>`   | Marks the beginning of the entire text.  |
| `<\| end_of_text \|>`     | Marks the end of the entire text.        |

### Template

The Hermes template is designed to structure the conversation in a way that the model can understand. It handles system
messages, user messages, assistant responses, and tool calls.

**Key Template Logic:**

* **System Messages and Tools:** If a system message or tools are provided, they are wrapped in `<|im_start|>system` and
  `<|im_end|>` tags. The template provides specific instructions for function calling.
* **Message Loop:** The template iterates through the messages, handling different roles:
    * **`tool`:** Tool responses are grouped under `<|im_start|>tool` and `<|im_end|>` tags.
    * **`user` and `assistant`:** Messages are wrapped in `<|im_start|>role` and `<|im_end|>` tags.
* **Tool Calls:** Assistant tool calls are formatted as a JSON object within `<tool_call>` tags.

```
{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{{- range .Tools }} 
{{ json . }}
{{- end }}
</tools>
For each function call return a JSON object, with the following pydantic model json schema for each:
{'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
Each function call should be enclosed within <tool_call> </tool_call> XML tags.
<tool_call>
{'name': <function-name>, 'arguments': <args-dict>}
</tool_call>
{{- end }}<|im_end|>
{{- end }}

{{- $hasToolResponses := false }}
{{- range .Messages }}
{{- if eq .Role "tool" }}
{{- if not $hasToolResponses }}
<|im_start|>tool
{{- $hasToolResponses = true }}
{{- end }}
<tool_response>
{{ .Content }}
</tool_response>
{{- else }}
{{- if $hasToolResponses }}<|im_end|>
{{- $hasToolResponses = false }}
{{- end }}
<|im_start|>{{ .Role }}
{{- if and (eq .Role "assistant") .ToolCalls }}
<tool_call>
{{- range .ToolCalls }}
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
</tool_call>
{{- else }}
{{ .Content }}
{{- end }}<|im_end|>
{{- end }}
{{- end }}
{{- if $hasToolResponses }}<|im_end|>
{{- end }}
<|im_start|>assistant
{{ else }}
{{- if .System }}
<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}

{{- if .Prompt }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}

<|im_start|>assistant
{{ .Response }}<|im_end|>
{{- end }}
```

## Llama 3.1

### Parameters

The `stop` parameter for Llama 3.1 is simpler than Hermes, using a different set of tokens to control generation.

| Token                   | Description                                                       |
|:------------------------|:------------------------------------------------------------------|
| `<\|start_header_id\|>` | Marks the beginning of a message header, which contains the role. |
| `<\|end_header_id\|>`   | Marks the end of a message header.                                |
| `<\|eot_id\|>`          | "End of Turn" token, marking the end of a message.                |

### Template

The Llama 3.1 template uses a header-based format to distinguish between different parts of the conversation.

**Key Template Logic:**

* **Headers:** Each message is preceded by `<|start_header_id|>role<|end_header_id|>` to define the role (system, user,
  assistant).
* **End of Turn:** Each message concludes with the `<|eot_id|>` token.
* **Tool Instructions:** When tools are present, the system message provides specific instructions on how to format the
  function call JSON.
* **Tool Calls:** Assistant tool calls are formatted as a JSON object with "name" and "parameters" keys.
* **Tool Responses:** Tool responses are identified with the `ipython` role in the header.

```
{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

Cutting Knowledge Date: December 2023

When you receive a tool call response, use the output to format an answer to the orginal user question.

You are a helpful assistant with tool calling capabilities.
{{- end }}<|eot_id|>
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
{{- if and $.Tools $last }}

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ range $.Tools }}
{{- . }}
{{ end }}
Question: {{ .Content }}<|eot_id|>
{{- else }}

{{ .Content }}<|eot_id|>
{{- end }}{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}
{{ range .ToolCalls }}
{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
{{- else }}

{{ .Content }}
{{- end }}{{ if not $last }}<|eot_id|>{{ end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}
{{- end }}
```
