import time

from flask import Flask, request, jsonify, render_template

from genai_flask_app.config import LLAMA_MODEL_ID
from genai_flask_app.custom_tools import tools_list
from genai_flask_app.model import get_ai_json_response, get_ai_tool_response
from genai_flask_app.structures.model_select import ModelSelection

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_message = data.get('message')
    model = data.get('model')

    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400

    start_time = time.time()

    try:
        model_selection_prompt = (
            "Select the model configuration that will efficiently fulfill the user's prompt. "
            "Based on the user's request, select the appropriate `model_id` and `mode`. "
            f"The user's request is: '{model}'\n"
            "You MUST reply with a valid JSON object with NO additional text, just the JSON. "
            "For example: {\"model_id\": \"tinyllama\", \"mode\": \"SIMPLE\"}"
        )

        model_config: ModelSelection = get_ai_json_response(
            LLAMA_MODEL_ID,
            model_selection_prompt,
            model,
            ModelSelection
        )
        model = model_config.model_id
        if model_config.mode == "TOOL_USAGE":
            return get_ai_tool_response(model, "You are a helpful assistant with tools for math and text formatting.", user_message, tools_list)
        elif model_config.mode == "FORMATTED":
            return get_ai_json_response(model, "You are a helpful assistant that formats answers into JSON.", user_message,)
        elif model_config.mode == "SIMPLE":
            # Handle SIMPLE mode
            pass

        app.logger.info("Model config: %s", model_config)

        response_data = model_config
        app.logger.info(f"Model config to be returned: {response_data}")
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.exception("Error generating response")
        return jsonify({"error": str(e)}), 500
    finally:
        duration = time.time() - start_time
        app.logger.info("Request duration: %.2f seconds", duration)


if __name__ == "__main__":
    app.run(debug=True)