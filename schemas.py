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
class Reflection(BaseModel):
    missing: str = Field(default="", description="Critica de lo que falta en la respuesta")
    superfluous: str = Field(default="", description="Que sobra o es irrelevante")

class AnswerQuestion(BaseModel):
    """Responde a la pregunta"""
    answer: str = Field(description="Respuesta detallada a la pregunta (unas 300 palabras).")
    reflection: Reflection = Field(default_factory=Reflection, description="Tu reflexion sobre la respuesta inicial")
    search_queries: List[str] = Field(default_factory=list, description="Entre 1 y 3 consultas de busqueda para investigar mjoras que respondan a la critica a tu respuesta actual.")

class ReviseAnswer(AnswerQuestion):
    """Revisa tu respuesta original a la pregunta."""
    references: List[str] = Field(default_factory=list, description="Referencias que motivan tu respuesta revisada.")