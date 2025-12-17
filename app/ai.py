from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = f"""
You are an AI tutor for teaching Python.
Explain concepts step by step.
Use simple language.
Always ask if the user wants an example.
Do not answer questions unrelated to tutoring.
"""
def get_ai_response(input_user: str) -> str:
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": input_user}
                    ]
                }
            ]
        )
        return response.output_text or ""
    except Exception as e:
        raise RuntimeError(f"AI Response failed: {e}")


        
# def get_embedding(text: str):
#     try:
#         response = client.embeddings.create(
#             model='text-embedding-3-small',
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         raise RuntimeError(f"AI Response failed: {e}")

    


