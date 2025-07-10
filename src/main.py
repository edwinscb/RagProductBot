import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.models import QueryRequest
from src.rag.vector_store import VectorStore, ProductDocument
from src.agents.responder_agent import ResponderAgent

app = FastAPI(
    title="Product-Query Bot API",
    description="Microservicio para responder preguntas sobre productos mediante un pipeline RAG."
)

DATA_PATH = os.getenv("DATA_PATH", "data/")
if not os.path.exists(DATA_PATH):
    print(f"Advertencia: La carpeta de datos '{DATA_PATH}' no existe. Asegúrate de que los archivos .txt estén ahí.")

vector_store = VectorStore(data_path=DATA_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Advertencia: La variable de entorno OPENAI_API_KEY no está configurada. El Responder Agent no funcionará.")
    responder_agent = None 
else:
    responder_agent = ResponderAgent(openai_api_key=OPENAI_API_KEY)

try:
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 3))
except ValueError:
    print("Advertencia: TOP_K_RETRIEVAL en .env no es un número válido. Usando 3 por defecto.")
    TOP_K_RETRIEVAL = 3

@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la aplicación FastAPI."""
    print("Iniciando indexación de documentos...")
    try:
        vector_store.initialize_and_index()
        print("Indexación de documentos completada con éxito.")
    except Exception as e:
        print(f"Error durante la indexación: {e}")
        raise Exception(f"No se pudo inicializar el vector store: {e}")

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Endpoint para recibir preguntas de usuarios sobre productos.
    Valida la entrada y redirige a la estructura multi-agente.
    """
    try:
        user_id = request.user_id
        user_query = request.query

        print(f"Received query from user '{user_id}': '{user_query}'")

        if not vector_store.index:
            raise HTTPException(status_code=503, detail="Vector store no inicializado. Intente de nuevo más tarde.")
               
        retrieved_documents: List[ProductDocument] = vector_store.retrieve(user_query, top_k=TOP_K_RETRIEVAL)
        retrieved_product_ids = [doc.product_id for doc in retrieved_documents]
        
        print(f"Documentos recuperados para '{user_query}': {retrieved_product_ids}")

        if responder_agent:
            bot_response_text = responder_agent.generate_response(user_query, retrieved_documents)
        else:
            bot_response_text = (
                "Lo siento, el Agente Respondedor no está configurado (falta la clave API de OpenAI). "
                "Pude recuperar información sobre los siguientes productos: "
                f"{', '.join(retrieved_product_ids) if retrieved_product_ids else 'ninguno'}."
            )

        final_response = {
            "user_id": user_id,
            "original_query": user_query,
            "bot_response": bot_response_text,
            "retrieved_product_ids": retrieved_product_ids
        }

        return final_response

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e.errors()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)