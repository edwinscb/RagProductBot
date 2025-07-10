import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from src.models import QueryRequest

app = FastAPI(
    title="Product-Query Bot API",
    description="Microservicio para responder preguntas sobre productos mediante un pipeline RAG."
)

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
        simulated_response = {
            "user_id": user_id,
            "original_query": user_query,
            "bot_response": f"Tu pregunta sobre '{user_query}' ha sido recibida. Procesando..."
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