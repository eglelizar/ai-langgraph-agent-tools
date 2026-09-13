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
    missing: str = Field(default="", description="Información faltante")
    superfluous: str = Field(default="", description="Información superflua o innecesaria")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="Respuesta principal redactada")
    reflection: Reflection = Field(default_factory=Reflection, description="Crítica de la respuesta")
    search_queries: List[str] = Field(default_factory=list, description="Consultas de búsqueda recomendadas")

class ReviseAnswer(BaseModel):
    answer: str = Field(description="Respuesta revisada y mejorada")
    reflection: Reflection = Field(default_factory=Reflection, description="Crítica de la revisión")
    search_queries: List[str] = Field(default_factory=list, description="Nuevas consultas de búsqueda")
    references: List[str] = Field(default_factory=list, description="Lista de URLs de referencias")