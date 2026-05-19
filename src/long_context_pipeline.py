def create_long_context_prompt(document, question):

    prompt = f"""
You are an intelligent AI assistant.

Use ONLY the provided document to answer the question.

Document:
{document}

Question:
{question}

Answer:
"""

    return prompt


def truncate_document(tokenizer, document, max_tokens=6000):

    tokens = tokenizer.encode(document)

    truncated_tokens = tokens[:max_tokens]

    truncated_document = tokenizer.decode(
        truncated_tokens,
        skip_special_tokens=True
    )

    return truncated_document