## "Build Your First GenAI Application The Right Way"

![demo](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/TA4LkoCK7YtPp_xbBH4asg/genai-app-demo.gif) 
_you'll build the application pictured above!_

Build a smart AI-powered web application using Flask and cutting-edge language models! This hands-on project will teach you to create intelligent applications that generate structured responses and leverage local AI tools. You'll learn to integrate Ollama, implement JSON parsing, and engineer prompts that deliver consistent, actionable results.

## Learning Objectives

By the end of this project, you will be able to:

-   **Develop** a Flask web application integrated with AI capabilities
-   **Utilize** the `langchain-ollama` library to interact with advanced language models locally
-   **Implement** LangChain's `JsonOutputParser` for structured AI outputs
-   **Apply** prompt engineering techniques for generating actionable JSON responses
-   **Compare** and **evaluate** different language models like Llama, and Mistral
-   **Enhance** your application with modular and reusable AI integration code

## How Large Language Models Work

Large Language Models are thinking machines that can understand and generate human language with incredible fluency. Unlike traditional software that follows rigid rules, LLMs learn patterns from vast amounts of text to develop an intuitive understanding of language, knowledge, and reasoning. Think of them as having read millions of books and learned to predict what comes next in any conversation.

## Tokenization

Every journey begins with breaking text into digestible pieces called tokens. The sentence "Hello world" becomes separate tokens that the model can process. Using techniques like Byte Pair Encoding, common letter combinations become single tokens, while rare words get split apart. Each token gets assigned a unique number, transforming human language into mathematical data the model can understand.

If you'd like to take a closer look at how different language models perform tokenization, you can play around with [Tiktokenizer](https://tiktokenizer.vercel.app/), an online tool built by [David Duong](https://duong.dev/) that's designed for visualizing tokenized text input!

## Embeddings

These token numbers are then transformed into vectors - lists of hundreds of decimal numbers that capture meaning in mathematical space. Words with similar meanings cluster together in this high-dimensional space, like "cat" and "dog" being closer than "cat" and "airplane."

To understand how embeddings group information, explore the [Nomic Atlas](https://atlas.nomic.ai/) map linked below. Every dot you see is an embedding of a Wikipedia biography and the coloured clusters reveal underlying connections between people across history.

## Attention is All You Need

Now comes the transformer's secret weapon: attention mechanisms. As the model processes each word, it simultaneously looks at every other word in the sentence to understand relationships and context. When processing "The cat sat on the mat," the model learns that "sat" relates most strongly to "cat" as the subject performing the action. Multiple attention heads work in parallel, each specializing in different types of relationships like grammar, meaning, or long-range dependencies.

## Transformer Layers

These attention mechanisms stack into layers. Information flows upward through dozens of layers, with each one building more sophisticated representations. Early layers might focus on basic grammar and word relationships, while deeper layers develop complex reasoning abilities and factual knowledge. To keep this flow stable, the model also passes forward the original input of each layer alongside the new transformations. This helps preserve important details and prevents information from being lost or distorted as it moves through many layers.

See the below LLM visualization tool created by [Brendan Bycroft](https://bbycroft.net/) to take a closer look at the layers of various GPT models.

## Next Token Prediction

The model's training objective is easy enough: predict the next word in a sequence. Given "The capital of France is," it learns to predict "Paris." But this simple task teaches remarkable complexity - grammar, facts, reasoning, and even creativity. The model learns that language follows patterns, and these patterns encode the structure of how we think!

The below tool built by [Alonso Allende](https://alonsosilvaallende.github.io/) explores next token prediction in a fun, interesting way. Try playing around with different text samples!

## Massive Scale Training

Training happens on an enormous scale that's hard to comprehend. The model processes trillions of words from books, articles, and websites, learning from the collective knowledge of the internet and beyond. Thousands of powerful computers work together for months, adjusting billions of parameters through backpropagation. Each time the model makes a wrong prediction, it slightly adjusts its internal connections to do better next time.

## Inference

During actual use, the trained model generates text one token at a time. It considers everything you've said, processes it through all its layers of understanding, and predicts the most appropriate next word. That word gets added to the conversation and fed back in to predict the following word, creating a chain of coherent thought. Temperature and sampling controls the trade-off between creativity versus consistency.

## Pulling It All Together

Through this process of learning from human text, these models develop something remarkably close to understanding. They learn not just to parrot back information, but to reason, create, and engage in sophisticated dialogue. While we're still trying to figure out the mysteries of how intelligence emerges from these mathematical structures, one thing is clear: we've created thinking partners that can augment human intelligence in extraordinary ways!

## Setting Up Your Development Environment

Before we dive into development, let's set up your project environment. This guide assumes you have Python and Ollama installed on your system.

## Step 1: Install Ollama

If you haven't already, download and install Ollama from [ollama.com](https://ollama.com). After installation, make sure the Ollama server is running.

You can pull a model to get started. We'll use `llama3` for this lab.

```bash
ollama pull llama3
```

## Step 2: Create Your Project Directory

Open your terminal and run:

```bash
mkdir genai_flask_app
cd genai_flask_app
```

This creates a new directory for your project and navigates into it.

## Step 3: Set Up a Python Virtual Environment

Initialize a new Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```
*Note: On Windows, the activation command is `venv\Scripts\activate`*

## Step 4: Install Required Libraries

With your virtual environment activated, install the necessary Python libraries:

```bash
pip install langchain langchain-community ollama Flask
```

This command installs the core libraries we'll use for our application.

Now that your environment is set up, you're ready to start building your GenAI application!

## Using the Ollama Python Library

Let's make our very first call to a local model using the `ollama` Python library.

Create the file `capital.py`:

Open File `genai_flask_app/capital.py`

Let's start by adding in imports:

```python
import ollama

```

This imports the required module to interact with the local Ollama server.

```python
response = ollama.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': 'Only reply with the answer. What is the capital of Canada?'}
    ]
)

print(response['message']['content'])
```

Let's break this down:
- `model='llama3'`: We specify which Ollama model to use.
- `messages`: We provide a list of messages. For a simple query, we use a single message with the `user` role.

Now, run the script:
```bash
python capital.py
```

Running this code you should get the expected answer:

> _Ottawa_

Great job - you've called your first local LLM!

## Trying Other LLMs

There are numerous LLMs available through Ollama, each with its own strengths.

## How to Choose the Right LLM

Choosing an LLM model is deceptively complicated. While it’s tempting to focus on specs alone—like token limits, training data, or number of parameters—the real test comes when you evaluate how a model performs for your specific use cases.

Here are some important factors to consider:
-   **Capabilities**: Does the model meet your needs? Some models are better at coding, others at creative writing.
-   **Speed**: How quickly does the model generate responses? This depends on the model size and your hardware.
-   **Quality**: How accurate and relevant are the model’s outputs for your tasks?

Ultimately, you’ll want to experiment to find the right fit.

Now let's try and update our code using a different LLM model, `mistral`. First, pull the model:

```bash
ollama pull mistral
```

Make sure you still have `capital.py` open:

Open File `genai_flask_app/capital.py`

Now simply update the model from `llama3` to `mistral`. The new code should look like the following:

```python
import ollama

response = ollama.chat(
    model='mistral',
    messages=[
        {'role': 'user', 'content': 'Only reply with the answer. What is the capital of Canada?'}
    ]
)

print(response['message']['content'])
```

Now run the code in the terminal again:

```bash
python capital.py
```

You should get a similar correct answer. The output might have slight variations in phrasing, which highlights how different models can have different "personalities".

## Prompting and Structured Outputs

For simple queries, a direct prompt is fine. But for more complex tasks, especially when you need a structured output like JSON, you need to be more specific in your prompting. This is often called "Prompt Engineering".

For example, instead of getting:

```
The capital of Canada is Ottawa.
```

we might want:

```json
{ "country": "Canada", "capital": "Ottawa" }
```

This kind of structure makes it much easier to integrate AI into applications.

Managing prompts, roles, and output formatting manually can get messy fast. That’s where **LangChain** comes in.

## What is LangChain?

LangChain provides an abstraction layer over multiple language models, allowing developers to use a consistent API and set of tools. It includes built-in utilities for managing prompts, chaining responses, parsing outputs, and structuring conversations, making it a powerful toolkit for building sophisticated AI applications.

# Why Use LangChain?

-   **Consistent and Modular Integration**: Modular and reusable components simplify the integration of AI models.
-   **Structured Outputs with JSON Parsers**: Help ensure that responses from the language model are consistent and easily parsed.
-   **Support for Multi-Step Workflows**: Allows you to create complex, multi-step workflows.

Using LangChain enables developers to build robust, efficient, and maintainable AI solutions by simplifying the management of model interactions and ensuring that outputs are structured and reliable.

## Creating Your Flask Application

Now let's create the backbone of your Flask application. We'll set up a basic structure that we'll enhance with AI capabilities.

*If you haven't already, install the required libraries:*
```bash
pip install Flask langchain langchain-community ollama
```

## Step 1: Create Your Main Application File

Create a new file named `app.py`:

Open File `genai_flask_app/app.py`

Add the following code to set up a basic Flask app:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    # This is where we'll add our AI logic later
    return jsonify({"message": "AI response will be generated here"})

if __name__ == '__main__':
    app.run(debug=True)

```

Let's break down this code:
-   We import necessary modules from Flask.
-   We create a Flask application instance.
-   We define a route `/generate` that will handle POST requests. This is where our AI logic will go.
-   For now, it returns a simple JSON response.
-   The `if __name__ == '__main__':` block ensures the Flask development server runs when we execute this file directly.

You've set up the foundation of your GenAI application.

## Integrating AI Models with LangChain

Now, let's integrate AI capabilities into your Flask application using LangChain and Ollama.

## Step 1: Create a Model Integration File

Let's create a file to handle our AI model integration. Create a new file named `model.py`:

Open File `genai_flask_app/model.py`

```python
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# Function to initialize a model
def initialize_model(model_name):
    return ChatOllama(model=model_name)

# Initialize models
llama_llm = initialize_model("llama3")
mistral_llm = initialize_model("mistral")

# Prompt template
prompt = PromptTemplate(
    template="""
    {system_prompt}
    
    {format_prompt}
    
    User query: {user_prompt}
    """,
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

def get_ai_response(model, system_prompt, user_prompt):
    chain = prompt | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'format_prompt': json_parser.get_format_instructions()
    })

# Model-specific response functions
def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_llm, system_prompt, user_prompt)
```

Let's break down this code:
1.  We import `ChatOllama` from `langchain_community`, our interface to local Ollama models.
2.  We define a Pydantic model `AIResponse` to specify the desired JSON structure for the AI's output.
3.  `JsonOutputParser` is used to automatically parse the model's string output into the Pydantic object.
4.  We initialize our `llama3` and `mistral` models.
5.  We create a single, flexible `PromptTemplate`. The prompt includes instructions for the AI, the JSON format instructions from the parser, and the user's query.
6.  The `get_ai_response` function creates a "chain" that pipes the formatted prompt into the model, and then pipes the model's output into the JSON parser.
7.  We create wrapper functions (`llama_response`, `mistral_response`) for convenience.

This modular approach allows for easy addition of new models or modification of existing ones.

## Enhancing Your Flask Application with AI Capabilities

Now that we have our AI models set up, let's integrate them into our Flask application.

## Step 1: Update Your Flask Application

Let's update `app.py` to use these AI capabilities:

Open File `genai_flask_app/app.py`

Update the content of `app.py` with:

```python
from flask import Flask, request, jsonify, render_template
from model import llama_response, mistral_response
import time

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
    
    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."
    
    start_time = time.time()
    
    try:
        if model == 'llama':
            result = llama_response(system_prompt, user_message)
        elif model == 'mistral':
            result = mistral_response(system_prompt, user_message)
        else:
            return jsonify({"error": "Invalid model selection"}), 400
        
        result['duration'] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

Let's break down the changes:
1.  We import our model-specific response functions.
2.  In the `/generate` route, we now expect JSON input with 'message' and 'model' fields.
3.  We add error handling for missing inputs.
4.  We use a try-except block to handle potential errors in AI processing.
5.  We measure and include the processing time in the response.

## Step 2: Create the Frontend

For the user interface, we'll use simple HTML, CSS, and JavaScript.

Create a `templates` directory for your HTML file:
```bash
mkdir templates
```

Create the file `templates/index.html`:

Open File `genai_flask_app/templates/index.html`

Update the content of `templates/index.html` with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body class="font-ibm-plex">
    <div class="app-container">
        <!-- Header -->
        <div class="header">
            <div class="header-content">
                <h1 class="header-title">AI Assistant</h1>
                <button id="clearBtn" class="clear-btn" style="display: none;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="m3 6 18 0"></path>
                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                        <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                    </svg>
                    Clear Chat
                </button>
            </div>
        </div>

        <!-- Chat Container -->
        <div class="chat-container">
            <!-- Messages Area -->
            <div class="messages-area">
                <div class="messages-content">
                    <!-- Welcome Screen -->
                    <div id="welcomeScreen" class="welcome-screen">
                        <div class="welcome-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
                                <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                                <line x1="9" y1="9" x2="9.01" y2="9"/>
                                <line x1="15" y1="9" x2="15.01" y2="9"/>
                            </svg>
                        </div>
                        <h2 class="welcome-title">Welcome to AI Assistant</h2>
                        <p class="welcome-text">Choose a model and start a conversation. I'm here to help with your questions and tasks.</p>
                    </div>

                    <!-- Messages Container -->
                    <div id="messagesContainer" class="messages-container"></div>

                    <!-- Loading Indicator -->
                    <div id="loadingIndicator" class="loading-indicator" style="display: none;">
                        <div class="loading-bubble">
                            <div class="loading-content">
                                <div class="loading-dots">
                                    <div class="dot"></div>
                                    <div class="dot"></div>
                                    <div class="dot"></div>
                                </div>
                                <span class="loading-text">AI is thinking...</span>
                            </div>
                        </div>
                    </div>

                    <!-- Messages End -->
                    <div id="messagesEnd"></div>
                </div>
            </div>

            <!-- Input Area -->
            <div class="input-area">
                <div class="input-content">
                    <form id="chatForm" class="chat-form">
                        <!-- Model Selection -->
                        <div class="model-section">
                            <span class="model-label">Model:</span>
                            <div class="select-wrapper">
                                <select id="modelSelect" class="model-select">
                                    <option value="llama">Llama</option>
                                    <option value="mistral">Mistral</option>
                                </select>
                            </div>
                        </div>

                        <!-- Message Input -->
                        <div class="input-section">
                            <div class="textarea-container">
                                <textarea 
                                    id="messageInput" 
                                    class="message-textarea"
                                    placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                                    rows="1"
                                ></textarea>
                            </div>
                            
                            <button type="submit" id="sendButton" class="send-button">
                                <svg id="sendIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <line x1="22" y1="2" x2="11" y2="13"></line>
                                    <polygon points="22,2 15,22 11,13 2,9 22,2"></polygon>
                                </svg>
                                <div id="loadingSpinner" class="loading-spinner" style="display: none;">
                                    <div class="spinner"></div>
                                </div>
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/script.js"></script>
</body>
</html>
```

## Step 3: Adding CSS and JavaScript

Create a `static` folder in your project and download the necessary CSS and JavaScript files.

```bash
mkdir static
```

```bash
wget -O static/script.js "https://gist.githubusercontent.com/tenzinmigmar/0168709391266a8d8da7936f1a866c71/raw/95f4f4e1a1966b3f5183dd2f822cfcfd08d2238a/script.js"
```

```bash
wget -O static/styles.css "https://gist.githubusercontent.com/tenzinmigmar/278575598f79a4940993a1fc8640a60a/raw/24eda98885e854b01b4a46d1756112e91d3acc10/styles.css"
```

## Step 4: Testing Your AI-Enabled Application

First, make sure your Ollama server is running. Then, run your Flask application:

```bash
python app.py
```

You should see output indicating that the Flask development server is running. Open your web browser and navigate to `http://127.0.0.1:5000`.

Try this with different messages and models to see how the responses vary.

Congratulations you've created your LLM-enabled Flask Application running entirely on your local machine!

## Conclusion and Next Steps

Congratulations on completing this guided project! You've successfully built a backend for a GenAI application using Flask, integrated multiple local AI models with Ollama, and implemented structured JSON outputs for enhanced functionality.

## Key Takeaways

-   You've learned to set up a Flask application with local AI capabilities.
-   You've integrated and compared multiple language models (Llama, Mistral) via Ollama.
-   You've implemented LangChain's `JsonOutputParser` for structured AI outputs.
-   You've gained insights into prompt engineering for local models.
-   You've created a modular and maintainable codebase for AI integration.

## Next Steps

To further enhance your skills and application:

1.  **Implement Caching**: Add a caching mechanism to improve performance for repeated queries.
2.  **Explore Advanced LangChain Features**: Look into features like memory for maintaining conversation context.
3.  **Add More Models**: Try integrating other models available through Ollama.
4.  **Enhance Error Handling**: Implement more robust error handling and logging.
5.  **Build a more complex frontend**: Use a framework like React or Vue to build a more dynamic user interface.

Remember, the field of GenAI is rapidly evolving. Keep experimenting, learning, and building to stay at the forefront of this exciting technology!

Thank you for participating in this workshop. We hope you found it valuable and are inspired to continue your journey in AI-driven application development!