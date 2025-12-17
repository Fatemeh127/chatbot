import gradio as gr
from ai import get_ai_response

def chat(message, history):
    if history is None:
        history = []

    history.append({"role": "user", "content": message})

    response = get_ai_response(message)
    history.append({"role": "assistant", "content": response})

    return history, "" 

with gr.Blocks(title="🐍 Python Tutor Chatbot") as demo:
    gr.Markdown(
        """
        # 🐍 Python Tutor Chatbot
        Welcome 👋  
        I am your AI Python tutor.  
        Ask me any Python-related question and I will explain it step by step.
        """
    )

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(
        placeholder="Type your question and press Enter...",
        lines=1
    )

    msg.submit(chat, [msg, chatbot], [chatbot, msg])

demo.launch(share=True)