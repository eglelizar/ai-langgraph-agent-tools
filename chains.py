"""Cadenas (chains) del agente: borrador inicial y revisión.

Este módulo construye:
- Un prompt “actor” que pide respuesta + reflexión + queries.
- Un `first_responder` que genera el primer borrador en formato `AnswerQuestion`.
- Un `revisor` que reescribe la respuesta y añade referencias con `ReviseAnswer`.
"""
# 1. IMPORTACIONES
#------------------------------------------------------------------------------------------------
import datetime  # Para inyectar la hora actual en el prompt (contexto temporal).

from dotenv import load_dotenv  # Para cargar variables de entorno (p.ej., claves de OpenAI/Tavily).

load_dotenv()  # Lee `.env` y variables del sistema para configurar credenciales.

from langchain_core.messages import HumanMessage  # Tipo de mensaje “humano” para inicializar conversaciones.
from langchain_core.output_parsers.openai_tools import (  # Parsers para interpretar tool calls del modelo.
    JsonOutputToolsParser,  # Parser genérico para extraer tool calls en JSON.
    PydanticToolsParser,  # Parser que valida y convierte args a modelos Pydantic.
)  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Plantillas de prompt y placeholder del historial.
from langchain_openai import ChatOpenAI  # Wrapper de LangChain para modelos chat de OpenAI.

from schemas import AnswerQuestion, ReviseAnswer  # Modelos Pydantic que definen el “formato requerido” del output.

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=600,
    request_timeout=20,  # Falla rápido si no hay respuesta en 20s
    temperature=0.7
)
 # Instancia del LLM; `model` selecciona el modelo a usar.
parser = JsonOutputToolsParser(return_id=True)  # Parser JSON; `return_id` conserva IDs de tool calls si existen.
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])  # Parser Pydantic para convertir la salida del tool `AnswerQuestion`.

# 2. PLANTILLAS DE PROMPT
#------------------------------------------------------------------------------------------------
actor_prompt_template = ChatPromptTemplate.from_messages(  
    [  
        (  
            "system",  # Rol del mensaje (system) para guiar el comportamiento del modelo.
            """Eres un investigador/a experto/a.
Hora actual: {time}

1. {first_instruction}
2. Reflexiona y critica tu respuesta. Sé severo/a para maximizar la mejora.
3. Recomienda consultas de búsqueda para investigar información y mejorar tu respuesta.""",  # Texto base del sistema con variables {time} y {first_instruction}.
        ), 
        MessagesPlaceholder(variable_name="messages"),  # Inserta el historial real del usuario/IA en `messages`.
        ("system", "Responde a la pregunta del usuario anterior usando el formato requerido."),  # Recordatorio: salida en formato del schema/tool.
    ]  
).partial(  # “Pre-rellena” variables del prompt (partial application).
    time=lambda: datetime.datetime.now().isoformat(),  # `time` se evalúa dinámicamente con la hora actual en ISO 8601.
)  


first_responder_prompt_template = actor_prompt_template.partial(  # Especialización del prompt para el primer borrador.
    first_instruction="Proporciona una respuesta detallada de máximo 300 palabras."  # Instrucción específica del paso 1 (borrador).
)  # Fin del partial para el primer respondedor.



revise_instructions = """Revisa tu respuesta anterior usando la nueva información.
    - Debes usar la crítica previa para añadir información importante a tu respuesta.
        - DEBES incluir citas numéricas en tu respuesta revisada para que se pueda verificar.
        - Añade una sección "Referencias" al final de tu respuesta (no cuenta para el límite de palabras). En formato:
            - [1] https://example.com
            - [2] https://example.com
    - Debes usar la crítica previa para eliminar información superflua de tu respuesta y ASEGURARTE de que no supere las 300 palabras.
"""  # Instrucciones del paso 2: reescritura con citas y control de longitud.

revisor_prompt_template = actor_prompt_template.partial(  # Reutiliza el mismo template “actor” para el revisor…
    first_instruction=revise_instructions  # …pero cambiando la instrucción inicial por reglas de revisión.
)

# 3. CREACIÓN DE CADENAS
#------------------------------------------------------------------------------------------------
first_responder = first_responder_prompt_template | llm.bind_tools( 
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"  # Fuerza al modelo a responder como tool con el esquema de datos `AnswerQuestion`.
)  

revisor = revisor_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer")  # Fuerza tool con el esquema de datos `ReviseAnswer` en la fase de revisión.

