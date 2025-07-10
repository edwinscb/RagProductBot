import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.models import QueryRequest
from src.vector_store import VectorStore, ProductDocument

app = FastAPI(
    title="Product-Query Bot API",
    description="Microservicio para responder preguntas sobre productos mediante un pipeline RAG."
)

DATA_PATH = os.getenv("DATA_PATH", "data/")
if not os.path.exists(DATA_PATH):
    print(f"Advertencia: La carpeta de datos '{DATA_PATH}' no existe. Asegúrate de que los archivos .txt estén ahí.")

vector_store = VectorStore(data_path=DATA_PATH)

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

        retrieved_documents: List[ProductDocument] = vector_store.retrieve(user_query, top_k=3)

        retrieved_content_snippets = [doc.content[:100] + "..." for doc in retrieved_documents]
        print(f"Documentos recuperados para '{user_query}': {retrieved_content_snippets}")

        if retrieved_documents:
            context_for_llm = "\n\n".join([doc.content for doc in retrieved_documents])
            bot_response_text = (
                f"Gracias por tu pregunta sobre '{user_query}'. "
                f"He encontrado información relevante sobre los productos: "
                f"{', '.join([doc.product_id for doc in retrieved_documents])}. "
                "En el futuro, aquí se generaría una respuesta basada en esta información."
            )
        else:
            bot_response_text = f"No pude encontrar información relevante para '{user_query}' en nuestra base de datos de productos."


        simulated_response = {
            "user_id": user_id,
            "original_query": user_query,
            "bot_response": bot_response_text,
            "retrieved_product_ids": [doc.product_id for doc in retrieved_documents]
        }


        return simulated_response

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e.errors()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)