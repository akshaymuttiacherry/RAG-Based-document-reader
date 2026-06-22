chat_history = []

def add_memory(question, answer):

    chat_history.append({
        "question": question,
        "answer": answer
    })

def get_memory():

    return chat_history