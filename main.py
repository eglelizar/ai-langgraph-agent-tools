import json
from langchain_core.messages import AIMessage, ToolMessage
from chains import first_responder, revisor  # Asegúrate de importar las cadenas sin el parser en chains.py

def draft_node(state: MessagesState):  
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
        
    return {"messages": [ai_msg]}