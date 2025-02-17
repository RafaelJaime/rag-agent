from os import getenv
import gradio as gr
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from llm import LLM, LLMModels
from EmailTool import EmailTool
from VectorStore import VectorStore
from TariffRagTool import TariffRagTool

load_dotenv()

DEBUG = getenv("DEBUG", "False").lower() in ("true", "1")
SYSTEM_PROMPT = (
    "Eres un asistente experto en regulaciones aduaneras y aranceles de Ecuador. "
    "Tu tarea es identificar el código arancelario de los objetos mencionados utilizando el contexto proporcionado. "
    "Si el usuario menciona un objeto, debes devolver el código arancelario más amplio disponible y su arancel aplicado. "
    "Si el código es demasiado general (tiene menos de 8 dígitos), limítate a preguntar al usuario por los que continúan a ese código "
    "(por ejemplo, si es 10.04, busca los códigos que comienzan con 1004...). "
    "Si el usuario proporciona una lista, devuelve la lista con sus códigos correspondientes y el arancel que se le aplica a cada uno. "
    "Si no encuentras el código exacto, haz preguntas para refinar la búsqueda, basándote en el contexto y tu conocimiento. "
    "Si la información no está disponible, indícalo claramente."
)

def _print_fragments(relevant_docs):
    print("\n=== Fragmentos de documento utilizados para la respuesta ===")
    for i, doc in enumerate(relevant_docs, 1):
        texto = doc.page_content.replace("\n", " ")
        print(f"\nFragmento {i}:\n{texto[:300]}...")

def chatbot(message, history):
    messages_for_agent = []
    for msg in history:
        if msg["role"] == "user":
            messages_for_agent.append(HumanMessage(content=msg["content"]))
        else:
            messages_for_agent.append(AIMessage(content=msg["content"]))
    messages_for_agent.append(HumanMessage(content=message))

    memory = MemorySaver()

    agent_executor = create_react_agent(
        LLM.chatOpenAIWithOpenRouter(LLMModels.OPENAI_GPTO3MINI), # We need to use models compatible with tools (deepseekr1 for example is not) https://huggingface.co/spaces/galileo-ai/agent-leaderboard
        tools=[TariffRagTool(), EmailTool()],
        checkpointer=memory
    )

    messages_for_agent.append(HumanMessage(content=message))

    streamed_history = []
    yield streamed_history

    config = {"configurable": {"thread_id": "demo-thread"}}

    for chunk in agent_executor.stream({"messages": messages_for_agent}, config=config):
        if "tools" in chunk:
            for tool_msg in chunk["tools"]["messages"]:
                tool_content = tool_msg.content

                if "TariffRagTool" in str(tool_msg.additional_kwargs.get("name", "")) or "tariff" in tool_content.lower():
                    thinking_content = (
                        "Buscando en los documentos...\n\n"
                        f"Encontrado:\n{tool_content}"
                    )
                    title = "Searching in official documents..."

                elif "EmailTool" in str(tool_msg.additional_kwargs.get("name", "")) or "email" in tool_content.lower():
                    thinking_content = (
                        "Preparando el envío de email...\n\n"
                        f"Detalles:\n{tool_content}"
                    )
                    title = "Enviando email..."
                else:
                    thinking_content = (
                        "Procesando con herramienta...\n\n"
                        f"Detalles:\n{tool_content}"
                    )
                    title = "Procesando..."

                thinking_msg = gr.ChatMessage(
                    role="assistant",
                    content=thinking_content,
                    metadata={"title": title}
                )
                streamed_history.append(thinking_msg)
                yield streamed_history

    if "agent" in chunk:
        for agent_msg in chunk["agent"]["messages"]:
            final_msg = gr.ChatMessage(
                role="assistant",
                content=agent_msg.content
            )
            streamed_history.append(final_msg)
            yield streamed_history


demo = gr.ChatInterface(
    chatbot,
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Escribe tu mensaje aquí...", container=False, scale=7),
    title="Asesor Aduanero - Aranceles y Regulaciones de Ecuador",
    description="Asesoría experta sobre los documentos, aranceles y regulaciones aduaneras en Ecuador.",
    examples=[
        "Quiero arancelar un cordero",
        "Quiero arancelar un cordero para reproduccion como reproductor de raza pura",
        "Mandar email a rajaimor@gmail.com con subject cuidado o no tanto"
    ],
    type="messages",
    editable=True,
    save_history=True,
)

if __name__ == "__main__":
    demo.queue().launch()