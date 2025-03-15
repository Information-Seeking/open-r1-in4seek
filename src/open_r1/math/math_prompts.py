def build_conv(question, prefix=""):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]
    if prefix != "":
        messages.append({
            "role": "assistant",
            "content": prefix
        })
    return messages