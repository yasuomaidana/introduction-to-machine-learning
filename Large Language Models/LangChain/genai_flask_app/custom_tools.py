from langchain_core.tools import Tool


def calculator(expression: str) -> str:
    """A simple calculator for mathematical expressions.
    Input  MUST be a simple math expression like '25 + 63' or '15 * 7'."""
    try:
        # HINT: Use Python's eval() function for simple calculations
        return f"Result: {eval(expression.strip().strip("\"'`"))}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input MUST be in the exact format: 'format_type: text'
    Examples: 'uppercase: hello world', 'titlecase: langchain is awesome'"""
    try:
        # Parse the input to get format type and text
        # Your code here
        if len(formatted := text.split(":")) > 1:
            format_type, content = formatted[0].strip(), formatted[1].strip().lower()
            if format_type == "uppercase":
                return content.upper()
            elif format_type == "lowercase":
                return content.lower()
            elif format_type == "titlecase":
                return content.title()
            else:
                return f"Unknown format type: {format_type}"
        else:
            return f"Missing format. Example: titlecase: {text} -> {text.title()}"
    except Exception as e:
        return f"Error formatting text: {str(e)}"


# Create Tool objects for our functions
# HINT: Use the Tool class to wrap the functions
tools_list = [
    Tool(name="Calculator", func=calculator,
         description="A simple calculator that can add, subtract, multiply, or divide two numbers."),
    Tool(name="Text Formatter", func=format_text,
         description="Format text to uppercase, lowercase, or title case.")
]