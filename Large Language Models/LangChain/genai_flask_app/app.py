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
            "You are an AI router. Analyze the user request and select the model, mode, and schema configuration "
            "that strictly adheres to the requested output format and task requirements."
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
            if hasattr(json_res, 'response'):
                response_text = json_res.response
            elif hasattr(json_res, 'model_dump'):
                response_text = str(json_res.model_dump())
            elif isinstance(json_res, dict):
                response_text = json_res.get('response', str(json_res))
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
