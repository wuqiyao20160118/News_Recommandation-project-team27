from nltk.tokenize import RegexpTokenizer


def tokenize_word(text):
    if not isinstance(text, str):
        return []
    tokenizer = RegexpTokenizer("[\w]+|[.,!?;|]")
    return tokenizer.tokenize(text)

