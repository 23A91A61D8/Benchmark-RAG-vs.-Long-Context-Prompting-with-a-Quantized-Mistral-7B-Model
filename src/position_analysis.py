def get_document_sections(tokenizer, document):

    tokens = tokenizer.encode(document)

    total_tokens = len(tokens)

    beginning = tokenizer.decode(
        tokens[:2000],
        skip_special_tokens=True
    )

    middle_start = total_tokens // 2

    middle = tokenizer.decode(
        tokens[middle_start:middle_start + 2000],
        skip_special_tokens=True
    )

    end = tokenizer.decode(
        tokens[-2000:],
        skip_special_tokens=True
    )

    return beginning, middle, end