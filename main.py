
#1. Importaciones

import json
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import first_responder, revisor  # Asegúrate de importar las cadenas sin el parser en chains.py
from tool_executor import execute_tools

MAX_ITERATIONS = 2

#2. Creacion de los nodos del grafo

def draft_node(state: MessagesState):    #Nodo BORRADOR
    """Genera el primer borrador de la respuesta."""
    response = first_responder.invoke({"messages": state["messages"]})  
    
    # Manejo defensivo por si el LLM devuelve contenido vacío o texto plano
    content_text = response.content if hasattr(response, "content") else str(response)
    try:
        # Intentamos verificar si ya es un JSON válido
        json.loads(content_text)
        ai_msg = AIMessage(content=content_text)
    except (json.JSONDecodeError, TypeError):
        # Si no es JSON, empaquetamos el texto dentro del esquema esperado en formato JSON string
        fallback_data = {
            "answer": content_text,
            "reflection": {"missing": "", "superfluous": ""},
            "search_queries": ["baterias estado solido startups retos"]
        }
        ai_msg = AIMessage(content=json.dumps(fallback_data))
        print ("Nodo BORRADOR respuesta: ", ai_msg)
    return {"messages": [ai_msg]}  


def revise_node(state: MessagesState):  
    """Revisa la respuesta basándose en los resultados de herramientas."""  
    response = revisor.invoke({"messages": state["messages"]})  
    
    content_text = response.content if hasattr(response, "content") else str(response)
    try:
        json.loads(content_text)
        ai_msg = AIMessage(content=content_text)
    except (json.JSONDecodeError, TypeError):
        fallback_data = {
            "answer": content_text,
            "reflection": {"missing": "", "superfluous": ""},
            "search_queries": [],
            "references": []
        }
        ai_msg = AIMessage(content=json.dumps(fallback_data))
        print ("Nodo REVISOR respuesta: ", ai_msg)
        
    return {"messages": [ai_msg]}


def event_loop (state: MessagesState)-> Literal ["execute_tools", END]:
    """Decide si continuar o finalizar sgun el numbero de interaciones"""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iteractions = count_tool_visits
    if (num_iteractions > MAX_ITERATIONS):
        return END
    return "execute_tools"

#----------->4. Creacion del grafo

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
    #print(response)

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

