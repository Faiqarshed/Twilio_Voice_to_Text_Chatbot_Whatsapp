from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max size of each chunk (tokens/characters depending on length_function)
    chunk_overlap=100,     # overlap to preserve context
    separators=["\n\n", "\n", ".", " ", ""]
)

