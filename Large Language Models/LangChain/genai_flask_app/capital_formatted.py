from typing import cast

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

from genai_flask_app.structures.country_with_capital import CountryWithCapital

output_parser = JsonOutputParser(pydantic_object=CountryWithCapital)
model = ChatOllama(model="llama3.2")
prompt = PromptTemplate(template="""Only reply with the answer.
{format_instructions}
What is the capital of {country_name}?""",
                        input_variables=["country_name"],
                        partial_variables={"format_instructions": output_parser.get_format_instructions()},
                        )

chain = prompt | model | output_parser
response = chain.invoke({"country_name": "Canada"})

print(cast(CountryWithCapital, response))
