"""Cadenas (chains) del agente: borrador inicial y revisión."""

import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=600,
    request_timeout=20,
    temperature=0.7
)

parser_answer = PydanticOutputParser(pydantic_object=AnswerQuestion)
parser_revise = PydanticOutputParser(pydantic_object=ReviseAnswer)

actor_prompt_template = ChatPromptTemplate.from_messages(  
    [  
        (  
            "system",  
            """Eres un investigador/a experto/a.
Hora actual: {time}

1. {first_instruction}
2. Reflexiona y critica tu respuesta. Sé severo/a para maximizar la mejora.
3. Recomienda consultas de búsqueda para investigar información y mejorar tu respuesta.

{format_instructions}""",  
        ), 
        MessagesPlaceholder(variable_name="messages"),  
    ]  
).partial(  
    time=lambda: datetime.datetime.now().isoformat(),  
)  

first_responder_prompt_template = actor_prompt_template.partial(  
    first_instruction="Proporciona una respuesta detallada de máximo 300 palabras.",
    format_instructions=parser_answer.get_format_instructions()
)  

revise_instructions = """Revisa tu respuesta anterior usando la nueva información.
    - Debes usar la crítica previa para añadir información importante a tu respuesta.
        - DEBES incluir citas numéricas en tu respuesta revisada para que se pueda verificar.
        - Añade una sección "Referencias" al final de tu respuesta (no cuenta para el límite de palabras). En formato:
            - [1] https://example.com
            - [2] https://example.com
    - Debes usar la crítica previa para eliminar información superflua de tu respuesta y ASEGURARTE de que no supere las 300 palabras.
"""  

revisor_prompt_template = actor_prompt_template.partial(  
    first_instruction=revise_instructions,
    format_instructions=parser_revise.get_format_instructions()
)

first_responder = first_responder_prompt_template | llm | parser_answer  
revisor = revisor_prompt_template | llm | parser_revise