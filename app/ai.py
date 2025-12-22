from openai import OpenAI
import faiss
import numpy as np

client = OpenAI()

SYSTEM_PROMPT = f"""
You are an AI tutor for teaching Python.
Explain concepts step by step.
Use simple language.
Always ask if the user wants an example.
Do not answer questions unrelated to tutoring.

Allowed knowledge:
- You may use general Python knowledge.
- You may use the retrieved context if provided.

Restrictions:
- Only answer questions related to Python.
- If a question is not about Python, politely refuse.
- Do not invent APIs or features that do not exist in Python.
- If unsure, say you are not sure.

"""

# Summarized Memory
MAX_SUMMARIES = 5  
conversation_summaries = []

def summarize_memory(new_message: str, role: str) -> str:
    """
   Using an LLM to summarize a new message based on previous summaries.
   The output is a short text that preserves the main context of the conversation.
    """
    current_summary = "\n".join(conversation_summaries)
    prompt = f"""
    You are an AI assistant helping to maintain a concise conversation summary.
    Current summaries:
    {current_summary}

    New message from {role}:
    {new_message}

    Update the summaries to keep only relevant information in a short, clear way.
    Return a one-sentence summary.
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    )
    summary = response.output_text.strip()
    return summary

def add_to_summary(new_message: str, role: str):
    summary = summarize_memory(new_message, role)
    conversation_summaries.append(summary)

    #limit conversation_summaries
    if len(conversation_summaries) > MAX_SUMMARIES:
        conversation_summaries.pop(0)

# embedding    
def get_embedding(text: str) -> list:
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    try:
        response = client.embeddings.create(
            model= "text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"AI response failed: {e}") from e
    
# chunck    
def chunk_text(text: str, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i+ chunk_size])
        chunks.append(chunk)

    return chunks

# vector DB
EMBEDDING_DIM = 1536
index = faiss.IndexFlatL2(EMBEDDING_DIM)
documents = []

def add_documents(texts: list[str]):
    for text in texts:
        emb = get_embedding(text)
        index.add(np.array([emb]).astype("float32"))
        documents.append(text)

# search
def retrieve_context(query: str, k=3) -> str:
    if index.ntotal == 0:
        return ""
    query_emb = get_embedding(query)
    distances, indices = index.search(np.array([query_emb]).astype("float32"), k)

    results = [documents[i] for i in indices[0]]
    return "\n".join(results)

# tutor function
def get_ai_response(input_user: str) -> str:
    try:
        context = retrieve_context(input_user)
        # guardrail
        messages = [{
            "role": "system",
             "content": [{
                "type": "input_text",
                "text": SYSTEM_PROMPT + "\n\n<context>\n" + context + "\n</context>"
            }]
        }]

        # add summaries to prompt
        for s in conversation_summaries:
            messages.append({
                "role": "system",
                "content": [{"type": "input_text", "text": f"Conversation summary: {s}"}]
            })

        # user message 
        messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": input_user}]
        })

        response = client.responses.create(

            model="gpt-4o-mini",
            input=messages
        )

        answer = response.output_text or ""

        # update summaries
        add_to_summary(input_user, "user")
        add_to_summary(answer, "assistant")

        return answer
    
    except Exception as e:
        raise RuntimeError(f"AI Response failed: {e}") from e
  
def main():
    python_text = """
    A for loop in Python is used to iterate over a sequence such as a list,
    tuple, string, or range. It allows executing a block of code multiple times.

    Example:
    for i in range(5):
        print(i)
    """

    chunks = chunk_text(python_text)
    add_documents(chunks)

    print(get_ai_response("Explain for loop in Python"))
    print(get_ai_response("Give me a simple example"))
    print(get_ai_response("Now combine it with if statement"))

    


