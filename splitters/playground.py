"""Compare text splitters without an LLM or API key.

Run: uv run --extra splitters python splitters/playground.py
NLTK setup: uv run --extra splitters python -m nltk.downloader -d .nltk_data punkt_tab
The token examples may download tokenizer data on their first run.
"""

import os
from pathlib import Path

import nltk
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    SpacyTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

TEXT = (
    "Loaders read files into documents. Documents retain source metadata.\n\n"
    "Splitters divide documents into smaller chunks. Overlap keeps some context "
    "between chunks.\n\n"
    "Embeddings represent each chunk as numbers. Retrievers find relevant chunks "
    "to help answer a question."
)


class FixedWidthTextSplitter(TextSplitter):
    """Custom splitter: exact character windows, even across word boundaries.

    Implementing split_text also provides create_documents and split_documents
    through TextSplitter. Unlike recursive splitting, whitespace is preserved.
    """

    def __init__(self, chunk_size: int = 60, chunk_overlap: int = 10) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        chunks = []
        step = self._chunk_size - self._chunk_overlap
        for start in range(0, len(text), step):
            chunks.append(text[start : start + self._chunk_size])
            if start + self._chunk_size >= len(text):
                break  # Do not emit an extra chunk containing only overlap.
        return chunks


def show_chunks(
    name: str, chunks: list[str] | list[Document], *, token_counts: bool = False
) -> None:
    """Print content, character lengths, and metadata for easy comparison."""
    print(f"\n=== {name}: {len(chunks)} chunk(s) ===")
    encoder = tiktoken.get_encoding("cl100k_base") if token_counts else None
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.page_content if isinstance(chunk, Document) else chunk
        size = f"{len(text)} characters"
        if encoder is not None:
            size += f", {len(encoder.encode(text))} tokens"
        print(f"\nChunk {index} ({size}): {text!r}")
        if isinstance(chunk, Document):
            print("metadata:", chunk.metadata)


def main() -> None:
    documents = [Document(page_content=TEXT, metadata={"source": "lesson.txt"})]

    # 1. Recursive: character budget, with source metadata retained.
    recursive = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    show_chunks("RecursiveCharacterTextSplitter", recursive.split_documents(documents))

    # 2. Character: paragraph boundaries; a paragraph can exceed chunk_size.
    character = CharacterTextSplitter(
        separator="\n\n", chunk_size=100, chunk_overlap=0
    )
    show_chunks("CharacterTextSplitter", character.split_text(TEXT))

    # 3. Token: use the same encoding for splitting and displayed token counts.
    token = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=20, chunk_overlap=4
    )
    show_chunks("TokenTextSplitter", token.split_text(TEXT), token_counts=True)

    # 4. Recursive with a token budget, instead of the default character budget.
    recursive_tokens = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=20, chunk_overlap=4
    )
    show_chunks(
        "RecursiveCharacterTextSplitter.from_tiktoken_encoder",
        recursive_tokens.split_text(TEXT),
        token_counts=True,
    )

    # 5. Markdown: split_text returns Documents with heading metadata.
    markdown = "# RAG guide\n\n## Loading\n\n" + TEXT + (
        "\n\n## Retrieval\n\nSearch for chunks related to the question."
    )
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")]
    )
    sections = markdown_splitter.split_text(markdown)
    show_chunks("MarkdownHeaderTextSplitter", sections)
    # Apply a size limit after grouping by heading, preserving heading metadata.
    show_chunks("Markdown sections + recursive splitting", recursive.split_documents(sections))

    # 6. HTML: retain section headings as metadata.
    html = """<html><body><h1>RAG guide</h1>
    <h2>Loading</h2><p>Load files and preserve their source metadata.</p>
    <h2>Retrieval</h2><p>Find relevant chunks for a question.</p>
    </body></html>"""
    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[("h1", "title"), ("h2", "section")]
    )
    show_chunks("HTMLHeaderTextSplitter", html_splitter.split_text(html))

    # 7. JSON: accepts dictionaries, rather than a JSON string.
    payload = {
        "guide": {
            "loading": {"description": "Read files", "output": "Documents with metadata"},
            "splitting": {"description": "Divide text", "output": "Smaller document chunks"},
            "retrieval": {"description": "Search stored chunks", "output": "Relevant context"},
        }
    }
    json_splitter = RecursiveJsonSplitter(max_chunk_size=120, min_chunk_size=60)
    show_chunks("RecursiveJsonSplitter", json_splitter.create_documents([payload]))

    # 8. Code: language-aware separators favor classes and functions.
    code = '''class Guide:
    def load(self):
        return "Load documents and preserve metadata."

    def split(self):
        return "Split each document into smaller chunks."


def retrieve(question):
    return "Find chunks related to " + question
'''
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=100, chunk_overlap=0
    )
    show_chunks("RecursiveCharacterTextSplitter.from_language", code_splitter.split_text(code))

    # 9. NLTK: needs punkt_tab, installed separately (no implicit downloads).
    nltk.data.path.insert(0, str(Path(__file__).resolve().parents[1] / ".nltk_data"))
    nltk_splitter = NLTKTextSplitter(chunk_size=100, chunk_overlap=0)
    try:
        nltk_chunks = nltk_splitter.split_text(TEXT)
    except LookupError:
        print("\n=== NLTKTextSplitter: missing sentence tokenizer ===")
        print("Run: uv run --extra splitters python -m nltk.downloader -d .nltk_data punkt_tab")
    else:
        show_chunks("NLTKTextSplitter", nltk_chunks)

    # 10. spaCy: the rule-based sentencizer needs no downloaded language model.
    spacy_splitter = SpacyTextSplitter(
        pipeline="sentencizer", chunk_size=100, chunk_overlap=0
    )
    show_chunks("SpacyTextSplitter", spacy_splitter.split_text(TEXT))

    # 11. Custom: compare exact windows with recursive word-aware boundaries.
    custom = FixedWidthTextSplitter(chunk_size=60, chunk_overlap=10)
    show_chunks("Custom FixedWidthTextSplitter", custom.split_documents(documents))


if __name__ == "__main__":
    # Keep tokenizer downloads in the workspace, rather than a global cache.
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".tiktoken_cache")
    )
    main()
