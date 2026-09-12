"""Nodo/ejecutor de herramientas (tools) para el grafo.

Este módulo define cómo el agente ejecuta búsquedas web a partir de las
`search_queries` que genera el LLM durante la reflexión/revisión.
"""
# 1. IMPORTACIONES
#------------------------------------------------------------------------------------------------
from dotenv import load_dotenv  # Carga variables de entorno (p.ej., claves de Tavily/OpenAI).
load_dotenv()  # Lee el fichero `.env` y/o variables del sistema para configurar credenciales.

from langchain_core.tools import StructuredTool  # Wrapper para exponer funciones Python como tools.
from langchain_tavily import TavilySearch  # Tool de búsqueda (Tavily) integrada con LangChain.
from langgraph.prebuilt import ToolNode  # Nodo preconstruido para ejecutar tool calls en el grafo.

from schemas import AnswerQuestion, ReviseAnswer # Esquemas del fichero schemas.py que definen los modelos Pydantic para las herramientas.

# 2. CREACIÓN DE HERRAMIENTAS (TOOLS)
#------------------------------------------------------------------------------------------------

tavily_tool = TavilySearch(max_results=5)  # Cliente Tavily: limita resultados por consulta a 5.


def run_queries(search_queries: list[str], **kwargs):  # recibe la lista de queries del schema; kwargs ignora extras del tool call.
    """Ejecuta las consultas generadas."""  
    return tavily_tool.batch([{"query": query} for query in search_queries])  # Ejecuta en lote las consultas: una petición por query.


execute_tools = ToolNode(  # Nodo LangGraph que ejecuta herramientas cuando el LLM las solicita.
    [  # Lista de tools soportadas por este nodo.
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),  # Registra run_queries con el nombre de la tool del primer paso.
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),  # Registra run_queries también para el paso de revisión.
    ]  
)  
