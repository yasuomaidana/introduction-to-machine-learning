import time

from flask import Flask, request, jsonify, render_template

from genai_flask_app.config import LLAMA_MODEL_ID, TINYLLAMA_MODEL_ID, HERMES_MODEL_ID
from genai_flask_app.custom_tools import tools_list
from genai_flask_app.model import get_ai_json_response, get_ai_tool_response, get_ai_response
from genai_flask_app.structures.model_select import ModelSelection, Schema, GeneralResponse, SentimentResponse, Mode, \
    ModelID

app = Flask(__name__)

# Map dropdown option values to backend ModelID strings
MODEL_MAPPING = {
    'llama': LLAMA_MODEL_ID,
    'llama3': LLAMA_MODEL_ID,
    'llama3.2': LLAMA_MODEL_ID,
    'tiny': TINYLLAMA_MODEL_ID,
    'tinyllama': TINYLLAMA_MODEL_ID,
    'hermes': HERMES_MODEL_ID,
    'hermes3:8b': HERMES_MODEL_ID
}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    app.logger.info("Received request: %s", data)
    user_message = data.get('message')
    requested_model = data.get('model')

    if not user_message:
        return jsonify({"error": "Missing message"}), 400

    start_time = time.time()

    try:
        model_selection_prompt = (
            "You are a strict request classifier and router. Analyze the user message and select the proper mode:\n\n"
            "Classification Priority:\n"
            "1. TOOL_USAGE: Select if the input contains math expressions (e.g. '6*4^3', '2+2', arithmetic), math questions, or text formatting commands. Set schema_type = null.\n"
            "2. FORMATTED: Select ONLY if the user explicitly requests JSON output, sentiment analysis, or classification. Set schema_type to 'SENTIMENT' or 'GENERAL'.\n"
            "3. SIMPLE: Select for standard Q&A and conversation. Set schema_type = null."
        )

        model_config: ModelSelection = get_ai_json_response(
            LLAMA_MODEL_ID,
            model_selection_prompt,
            user_message,
            ModelSelection
        )
        app.logger.info("Model config: %s", model_config)

        # If user explicitly requested a model (and not 'auto'), respect their selection
        if requested_model and requested_model != 'auto' and requested_model in MODEL_MAPPING:
            model = MODEL_MAPPING[requested_model]
        else:
            model = model_config.model_id.value if isinstance(model_config.model_id, ModelID) else model_config.model_id

        # Determine mode
        mode = model_config.mode.value if isinstance(model_config.mode, Mode) else model_config.mode

        if mode == Mode.TOOL_USAGE or mode == "TOOL_USAGE":
            res = get_ai_tool_response(
                model,
                "You are a helpful assistant with tools for math and text formatting.",
                user_message,
                tools_list
            )
            response_text = res.get("output", str(res)) if isinstance(res, dict) else str(res)

        elif mode == Mode.FORMATTED or mode == "FORMATTED":
            schema_type = model_config.schema_type.value if isinstance(model_config.schema_type,
                                                                       Schema) else model_config.schema_type
            if schema_type == Schema.SENTIMENT or schema_type == "SENTIMENT":
                rt = SentimentResponse
            else:
                rt = GeneralResponse

            json_res = get_ai_json_response(
                model,
                "You are a helpful assistant that formats answers into JSON.",
                user_message,
                rt
            )
            if hasattr(json_res, 'model_dump_json'):
                response_text = json_res.model_dump_json(indent=2)
            elif hasattr(json_res, 'model_dump'):
                import json
                response_text = json.dumps(json_res.model_dump(), indent=2)
            elif isinstance(json_res, dict):
                import json
                response_text = json.dumps(json_res, indent=2)
            else:
                response_text = str(json_res)

        elif mode == Mode.SIMPLE or mode == "SIMPLE":
            response_text = get_ai_response(model, "You are a helpful assistant.", user_message)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        duration = time.time() - start_time
        return jsonify({
            "response": response_text,
            "duration": duration,
            "model_used": model,
            "mode_used": str(mode)
        })

    except Exception as e:
        app.logger.exception("Error generating response")
        return jsonify({"error": str(e)}), 500
    finally:
        duration = time.time() - start_time
        app.logger.info("Request duration: %.2f seconds", duration)


if __name__ == "__main__":
    app.run(debug=True)
