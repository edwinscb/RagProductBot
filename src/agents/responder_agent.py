import os
from typing import List
from openai import OpenAI
from src.rag.vector_store import ProductDocument 

class ResponderAgent:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)       
        self.model = "gpt-3.5-turbo" 

    def _build_prompt(self, query: str, context: List[ProductDocument]) -> str:
        """
        Construye el prompt para el LLM.
        Es CRUCIAL instruir al LLM para que use SÓLO el contexto proporcionado.
        """
        context_str = "\n---\n".join([doc.content for doc in context])

        prompt = f"""Eres un asistente de atención al cliente para una empresa de tecnología llamada
                     OrionTech.Tu objetivo es responder a las preguntas de los usuarios sobre los productos
                     de OrionTech basándote EXCLUSIVAMENTE en la información proporcionada en los documentos
                     de contexto.Si la pregunta no puede ser respondida con la información dada, o si la
                     información es contradictoria, responde que no tienes suficiente información para
                     responder a esa pregunta.NO inventes información. SÉ conciso y directo.

                    Documentos de Contexto:
                    ---
                    {context_str}
                    ---

                    Pregunta del usuario: "{query}"

                    Tu respuesta (basada solo en los Documentos de Contexto):
                    """
        return prompt

    def generate_response(self, query: str, retrieved_documents: List[ProductDocument]) -> str:
        """
        Genera una respuesta utilizando un LLM, basada en la consulta y los documentos recuperados.
        """
        if not retrieved_documents:
            return "Lo siento, no pude encontrar información relevante en nuestra base de datos de productos para responder a tu pregunta."

        prompt = self._build_prompt(query, retrieved_documents)

        print("--- Enviando prompt al LLM ---")
        print(prompt)
        print("-----------------------------")

        try:
            
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            response_content = chat_completion.choices[0].message.content
            return response_content
        except Exception as e:
            print(f"Error al llamar al LLM: {e}")
            return "Lo siento, hubo un problema al generar la respuesta. Por favor, inténtalo de nuevo más tarde."