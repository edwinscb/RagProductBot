import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ProductDocument:
    def __init__(self, product_id: str, content: str):
        self.product_id = product_id
        self.content = content
        self.embedding = None

    def to_dict(self):
        return {"product_id": self.product_id, "content": self.content}

class VectorStore:
    def __init__(self, data_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.model_name = model_name
        self.model = None
        self.documents: List[ProductDocument] = []
        self.index = None
        self.id_to_doc_map: Dict[str, ProductDocument] = {}

    def load_embedding_model(self):
        """Carga el modelo de embedding."""
        print(f"Cargando modelo de embedding: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Modelo de embedding cargado.")

    def load_documents(self):
        """Carga los documentos de productos desde la carpeta de datos."""
        print(f"Cargando documentos desde {self.data_path}...")
        for filename in os.listdir(self.data_path):
            if filename.endswith(".txt"):
                product_id = os.path.splitext(filename)[0] 
                filepath = os.path.join(self.data_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = ProductDocument(product_id=product_id, content=content)
                self.documents.append(doc)
                self.id_to_doc_map[product_id] = doc
        print(f"Cargados {len(self.documents)} documentos.")

    def generate_embeddings(self):
        """Genera embeddings para todos los documentos cargados."""
        if not self.model:
            raise ValueError("El modelo de embedding no ha sido cargado. Llama a 'load_embedding_model()' primero.")
        if not self.documents:
            raise ValueError("No hay documentos cargados. Llama a 'load_documents()' primero.")

        print("Generando embeddings para los documentos...")
        contents = [doc.content for doc in self.documents]
        embeddings = self.model.encode(contents, show_progress_bar=True)
        
        for i, doc in enumerate(self.documents):
            doc.embedding = embeddings[i]
        print("Embeddings generados.")

    def build_faiss_index(self):
        """Construye un índice FAISS a partir de los embeddings."""
        if not self.documents or self.documents[0].embedding is None:
            raise ValueError("Los embeddings no han sido generados. Llama a 'generate_embeddings()' primero.")

        print("Construyendo índice FAISS...")
       
        embeddings_matrix = np.array([doc.embedding for doc in self.documents]).astype('float32')
        dimension = embeddings_matrix.shape[1] 

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_matrix) 
        print(f"Índice FAISS construido con {self.index.ntotal} vectores.")

    def initialize_and_index(self):
        """Flujo completo para cargar datos, generar embeddings y construir el índice."""
        self.load_embedding_model()
        self.load_documents()
        self.generate_embeddings()
        self.build_faiss_index()
        print("Indexación completada.")

    def retrieve(self, query: str, top_k: int = 3) -> List[ProductDocument]:
        """
        Busca documentos relevantes para una consulta dada.
        Retorna los documentos ProductDocument completos.
        """
        if not self.index or not self.model:
            raise ValueError("El índice o el modelo de embedding no están inicializados.")

        query_embedding = self.model.encode([query]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_docs = []
        for idx in indices[0]: 
            if idx != -1: 
                retrieved_docs.append(self.documents[idx])
        return retrieved_docs