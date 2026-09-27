"""Cadenas (chains) del agente: borrador inicial y revisión."""


#1. Imports
import datetime
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticToolsParser

load_dotenv()

from langchain_core.messages import HumanMessage  #tipo de mensaje humano para inicializar contexto
from langchain_core.output_parsers import (
    JsonOutputToolsParser, 
    PydanticOutputParser
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#Si tuvieramos OpenAI tokens 
#from langchan_openai import chatOpenAI
from langchain_groq import ChatGroq

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

parser = JsonOutputToolsParser(return_id = True) #Parser JSON, conserva IDs de tool calls si existen
parser_pydentic = PydanticToolsParser(tools=[AnswerQuestion])

parser_answer = PydanticOutputParser(pydantic_object=AnswerQuestion)
parser_revise = PydanticOutputParser(pydantic_object=ReviseAnswer)


#2. Plantillas de PROMPT 

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
        MessagesPlaceholder(variable_name="messages"), ("system", "Responde a la pregutna del usuario anterior usando el formato requerido.") 
    ]  
).partial(  
    time=lambda: datetime.datetime.now().isoformat(),  
)  

first_responder_prompt_template = actor_prompt_template.partial(  
    first_instruction="Proporciona una respuesta concisa de máximo 150 palabras, en texto plano sin viñetas complejas.",
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

#3. Creacion de cadenas
# first_responder = first_responder_prompt_template | llm.bind_tools( tools = [AnswerQuestion], tool_choice = 'AnswerQuestion')
# revisor = revisor_prompt_template | llm.bind_tools( tools = [ReviseAnswer], tool_choice = 'ReviseAnswer')
# 3. Creacion de cadenas con structured output optimizado para Groq
first_responder = first_responder_prompt_template | llm.with_structured_output(AnswerQuestion, method="json_mode")
revisor = revisor_prompt_template | llm.with_structured_output(ReviseAnswer, method="json_mode")