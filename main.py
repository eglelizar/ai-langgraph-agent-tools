"""Grafo LangGraph del "Reflexion Agent”.

Este script construye un grafo con 3 nodos:
- `borrador`: genera un borrador con reflexión + queries.
- `execute_tools`: ejecuta búsquedas (Tavily) con esas queries.
- `revisor`: revisa la respuesta incorporando la info encontrada y añade referencias.
Luego itera un número limitado de veces y muestra el resultado.
"""
# 1. IMPORTACIONES
#------------------------------------------------------------------------------------------------
import json  # Para parsear args como string en tool_calls si es necesario.
from typing import Literal  # Tipado: para restringir el retorno de la función de enrutado.

from langchain_core.messages import AIMessage, ToolMessage  # Tipos de mensajes para inspección del output.
from langgraph.graph import END, START, StateGraph, MessagesState  # Primitivas de LangGraph: nodos, estado (sin necesidad de definir manualmente MessageGraph) y constantes.

from chains import revisor, first_responder  # Chains del LLM: borrador y revisor (definidas en `chains.py`).
from tool_executor import execute_tools  # Nodo ToolNode que ejecuta herramientas (definido en `tool_executor.py`).

MAX_ITERATIONS = 2  # Límite de vueltas de “buscar → revisar” (control de coste/tiempo).

# 2. CREACIÓN DE LOS NODOS DEL GRAFO
#------------------------------------------------------------------------------------------------
def draft_node(state: MessagesState):  # Nodo del grafo que produce el primer borrador.
    """Genera el primer borrador de la respuesta."""
    response = first_responder.invoke({"messages": state["messages"]})  # Llama al LLM usando el historial actual.
    return {"messages": [response]}  # Devuelve un delta de estado: añade el mensaje generado al historial.


def revise_node(state: MessagesState):  # Nodo del grafo que revisa tras ejecutar herramientas.
    """Revisa la respuesta basándose en los resultados de herramientas."""  
    response = revisor.invoke({"messages": state["messages"]})  # Llama al revisor con todo el historial (incluyendo ToolMessages).
    return {"messages": [response]}  # Devuelve un delta de estado: añade el mensaje de revisión al historial.


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:  # Función de enrutado: decide si se repite el bucle o se termina.
    """Decide si continuar o finalizar según el número de iteraciones.""" 
    count_tool_visits = sum(  # Cuenta cuántas veces se han añadido mensajes de herramienta al historial…
        isinstance(item, ToolMessage) for item in state["messages"]  # …detectando `ToolMessage` dentro de `messages`.
    ) 
    num_iterations = count_tool_visits  # En este diseño, una visita a tool equivale a una iteración del bucle.
    if num_iterations > MAX_ITERATIONS:  # Si superamos el máximo, cortamos el grafo.
        return END  # Señal de finalización para LangGraph.
    return "execute_tools"  # Si no, volvemos a ejecutar tools (buscar más) antes de revisar de nuevo.

# 4. CREACIÓN DEL GRAFO
#------------------------------------------------------------------------------------------------
builder = StateGraph(MessagesState)  # Crea el “builder” del grafo usando `MessagesState` como estado.
builder.add_node("borrador", draft_node)  # Registra el nodo `borrador` y su función.
builder.add_node("execute_tools", execute_tools)  # Registra el nodo que ejecuta herramientas.
builder.add_node("revisor", revise_node)  # Registra el nodo `revisor` y su función.
builder.add_edge(START, "borrador")  # Define el inicio: START → borrador, equivale a set_entry_point.
builder.add_edge("borrador", "execute_tools")  # Flujo: borrador → execute_tools.
builder.add_edge("execute_tools", "revisor")  # Flujo: execute_tools → revisor.
builder.add_conditional_edges("revisor", event_loop, ["execute_tools", END])  # Tras revise, elige: repetir tools o finalizar.
graph = builder.compile()  # Compila el grafo a un ejecutable.

print(graph.get_graph().draw_mermaid())  # Imprime un diagrama Mermaid del grafo (útil para depurar/entender).

if __name__ == "__main__":  

        # Entrada de ejemplo: un mensaje humano pidiendo mejorar una publicación
    inputs = {  
            "messages": [  # Historial inicial: un único mensaje del usuario.
                {  # Mensaje en formato dict (LangChain lo normaliza internamente).
                    "role": "user",  # Rol del mensaje como Human Message (user)
                    "content": "Escribe sobre la complejidad de utilizar baterías de estado sólido en coches eléctricos y lista startups que hayan desarrollado soluciones",  # Prompt del usuario.
                } 
            ] 
        }

    # Invocamos el grafo completo con la entrada
    response = graph.invoke(inputs)
 
    # OPCIÓN 1: (debug) mostrar todo el estado final
    # print(response)

    # OPCIÓN 2: Extraer SOLO los args del ÚLTIMO `ReviseAnswer`
    # (answer + reflection + search_queries + references)
    final_args = None
    for msg in reversed(response["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                # Nos interesa específicamente la última llamada a la tool `ReviseAnswer`
                if name != "ReviseAnswer":
                    continue
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                if isinstance(args, dict) and "answer" in args:
                    final_args = args
                    break
        if final_args is not None:
            break

    if final_args:
        print("--- RESPUESTA FINAL ---\n")
        # answer
        print(final_args.get("answer", ""))
        print("reflection:")
        reflection = final_args.get("reflection", {}) or {}
        print(f"  missing: {reflection.get('missing', '')}")
        print(f"  superfluous: {reflection.get('superfluous', '')}")

        # search_queries
        print("search_queries:")
        for q in final_args.get("search_queries", []) or []:
            print(f"  - {q}")

        # references (solo en ReviseAnswer)
        refs = final_args.get("references")
        if refs:
            print("references:")
            for r in refs:
                print(f"  - {r}")
    else:
        print("No se pudo extraer la respuesta final del revisor.")
