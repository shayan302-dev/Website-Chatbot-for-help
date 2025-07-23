import os
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=2000,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Session-based memory (resets on clear/refresh)
class SessionMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory()
    
    def clear(self):
        self.memory.clear()

session = SessionMemory()

# Chat functionality
def respond(message, chat_history):
    # Detect name
    if "my name is" in message.lower():
        name = message.split("my name is")[-1].strip()
        session.memory.save_context(
            {"input": "User shared their name"},
            {"output": f"Understood, your name is {name}"}
        )
    
    # Generate response
    conversation = ConversationChain(
        llm=ChatHuggingFace(llm=llm),
        memory=session.memory,
        verbose=False
    )
    response = conversation.predict(input=message)
    
    # Update chat history
    chat_history.append((message, response))
    return "", chat_history  # Clear input box

def clear_all():
    session.clear()  # Reset memory
    return [], ""    # Clear chat and input

# UI Design
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# <center>🤖 Shayyan's Chatbot</center>")
    
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message...", show_label=False, scale=7)
        send_btn = gr.Button("Send", variant="primary", scale=1)
    
    clear_btn = gr.Button("Clear Chat (Full Reset)")
    
    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_all, outputs=[chatbot, msg])

# Launch (simplified for Spaces)
app.launch()