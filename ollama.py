import ollama

def generate_answer(context, question):

    prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]