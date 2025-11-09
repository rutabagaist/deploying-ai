from main import get_graph
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv
import os

from logger import get_logger

_logs = get_logger(__name__)

llm = get_graph()

load_dotenv("../.env")
load_dotenv('../.secrets')

def travel_chat(message: str, history: list[dict]) -> str:
    langchain_messages = []
    n = 0
    _logs.debug(f"History: {history}")
    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
            n += 1
    langchain_messages.append(HumanMessage(content=message))

    state = {
        "messages": langchain_messages,
        "llm_calls": n
    }

    response = llm.invoke(state)
    return response['messages'][len(response['messages']) - 1].content

travel_assistant = gr.ChatInterface(
    fn = travel_chat,
    title = "Your humble travel assistant.",
    description = "I'm a chatbot assistant and I can help you find places to take photos in a new city, provide current weather conditions and give you a book recommendation that related to your current city. Just let me know where you are or where you are planning to travel to!",
    type = "messages",
    chatbot = gr.Chatbot(placeholder = "Hey how's it going, let me know where you are or where you'll travelling to and I'll help you with photo locations, tell you the current weather and offer a book recommendation!")
)

if __name__ == "__main__":
    _logs.info('Starting The Travel Chat App...')
    travel_assistant.launch()