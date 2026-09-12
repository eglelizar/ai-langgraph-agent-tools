"""Esquemas Pydantic usados como “tools” en el agente.

Este fichero define los modelos que LangChain/LangGraph usan para:
- Validar la salida del LLM cuando responde como herramienta (tool call).
- Estructurar la reflexión (crítica) y las consultas de búsqueda.
- En el nodo revisor, incluir referencias/citas para sustentar la revisión.
"""
# 1. IMPORTACIONES
#------------------------------------------------------------------------------------------------
from typing import List  # Tipado: lista de strings para queries y referencias.

# Pydantic es una biblioteca de Python que facilita la creación de modelos de datos robustos,
# usando validación automática y tipado estático basado en anotaciones de tipo. 
# Permite definir esquemas (modelos) para estructuras de datos complejas y garantiza su validez en tiempo de ejecución.
# BaseModel es la clase base de todo modelo Pydantic; Field permite añadir metadatos, descripciones y validaciones personalizadas a cada campo.
from pydantic import BaseModel, Field


# 2. MODELOS DE DATOS (ESQUEMAS PYDANTIC)
#------------------------------------------------------------------------------------------------
class Reflection(BaseModel):  # Modelo anidado con la crítica del borrador inicial.
    missing: str = Field(description="Crítica de lo que falta en la respuesta.")  
    superfluous: str = Field(description="Qué sobra o es irrelevante")


class AnswerQuestion(BaseModel):  # Esquema principal: respuesta + reflexión + queries de mejora.
    """Responde a la pregunta."""  # Docstring que se expone al LLM como intención del tool schema.

    answer: str = Field(description="Respuesta detallada a la pregunta (unas 300 palabras).") 
    reflection: Reflection = Field(description="Tu reflexión sobre la respuesta inicial.") # Con los campos de la clase anterior Reflection.
    search_queries: List[str] = Field(  # Lista de consultas que luego ejecutará Tavily.
        description="Entre 1 y 3 consultas de búsqueda para investigar mejoras que respondan a la crítica a tu respuesta actual."  # Guía para el LLM: 1–3 queries.
    ) 

class ReviseAnswer(AnswerQuestion):  # Hereda y extiende AnswerQuestion para la fase de revisión.
    """Revisa tu respuesta original a la pregunta."""  

    references: List[str] = Field(  # Lista de URLs o citas que justifican cambios en la respuesta.
        description="Referencias que motivan tu respuesta revisada."  
    ) 
