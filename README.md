# Product-Query Bot via RAG Pipeline

## Objetivo y Descripción

Este microservicio implementa un bot de consulta de productos utilizando un pipeline de Recuperación Aumentada por Generación (RAG). Actúa como un asistente de atención al cliente, respondiendo preguntas sobre productos basándose **exclusivamente** en un corpus de documentos internos, utilizando una arquitectura de dos agentes: un Agente Recuperador y un Agente Respondedor.

## Requisitos Clave Cumplidos

- **Endpoint POST /query**: Recibe consultas JSON (`user_id`, `query`).
- **Pipeline RAG**: Indexación de documentos (FAISS), recuperación `top-k` semántica y generación de respuestas con LLM (OpenAI) fundamentadas en el contexto.
- **Estructura Multi-Agente**: Separación de responsabilidades entre el Agente Recuperador (`VectorStore`) y el Agente Respondedor (`ResponderAgent`).
- **Calidad del Código**: Uso de variables de entorno para configuración (`.env`), modularidad.
- **Contenedorización**: `Dockerfile` para empaquetar y ejecutar la aplicación.

## Arquitectura

La aplicación se compone de:

- **Agente Recuperador (`VectorStore`)**: Encargado de cargar documentos (`.txt` de `data/`), generar sus embeddings (Sentence Transformers) e indexarlos en FAISS. Recupera los `top-k` documentos más relevantes para una consulta.
- **Agente Respondedor (`ResponderAgent`)**: Recibe la consulta del usuario y los documentos recuperados. Construye un prompt detallado para el LLM (OpenAI GPT-3.5 Turbo), instruyéndolo a responder solo con la información provista, y genera la respuesta final.

## Configuración y Ejecución

### 1. Requisitos Previos

- Python 3.9+
- pip
- Docker

### 2. Configuración Inicial

1.  **Clonar el Repositorio:**
    ```bash
    git clone [URL_DE_TU_REPOSITORIO]
    cd RagProductBot
    ```
2.  **Crear Entorno Virtual e Instalar Dependencias:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Configurar `.env`:**
    Crea un archivo `.env` en la raíz del proyecto. **¡Reemplaza `tu_clave_api_de_openai_aqui` con tu clave API real!**
    ```dotenv
    DATA_PATH=./data
    OPENAI_API_KEY=tu_clave_api_de_openai_aqui
    TOP_K_RETRIEVAL=3
    TOKENIZERS_PARALLELISM=false
    ```
4.  **Documentos de Productos:**
    Coloca tus archivos `.txt` con las descripciones de los productos en la carpeta `data/`.

### 3. Ejecutar la Aplicación

#### a) Ejecución Directa (Desarrollo)

1.  Activa tu entorno virtual.
2.  Ejecuta:
    ```bash
    python -m src.main
    ```
    El servidor se iniciará en `http://0.0.0.0:8000`.

#### b) Ejecución con Docker

1.  **Construir la Imagen:** (En la raíz del proyecto con `Dockerfile` y `data/`)
    ```bash
    docker build -t rag-product-bot .
    ```
2.  **Ejecutar el Contenedor:**
    Para ejecutar en segundo plano (recomendado):
    ```bash
    docker run -d -p 8000:8000 --name my-rag-bot rag-product-bot
    ```
    Para ver los logs en la terminal (se detendrá al cerrar):
    ```bash
    docker run -p 8000:8000 --name my-rag-bot rag-product-bot
    ```
3.  **Detener el Contenedor:**
    ```bash
    docker stop my-rag-bot
    # Opcional: docker rm my-rag-bot
    ```

## Cómo Probar (Ejemplo `curl`)

Una vez que el servidor esté corriendo, envía consultas:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{ \"user_id\": \"test_user_123\", \"query\": \"¿Cuáles son las características principales del Oriontech AuraBook Pro 15?\" }'
```
