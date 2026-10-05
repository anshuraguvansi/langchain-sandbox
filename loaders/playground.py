"""Load sample documents without an LLM or API key.

Run from the repository root:
    uv run --extra loaders python loaders/playground.py

The web example loads https://example.com and requires internet access.
The first seven loaders use the sunset langchain-community package and emit
a deprecation warning; see README.md for package status and migration guidance.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

DATA_DIR = Path(__file__).resolve().parent / "data"


def show_documents(name: str, documents: list[Document]) -> None:
    """Show the common output format shared by every loader."""
    print(f"\n=== {name}: {len(documents)} document(s) ===")
    for index, document in enumerate(documents, start=1):
        print(f"\nDocument {index}")
        print("page_content:", document.page_content.strip())
        print("metadata:", document.metadata)


def main() -> None:
    # 1. TextLoader: the entire text file becomes one document.
    text_loader = TextLoader(str(DATA_DIR / "intro.txt"), encoding="utf-8")
    show_documents("TextLoader", text_loader.load())

    # 2. PyPDFLoader: explicitly load one document per PDF page.
    pdf_loader = PyPDFLoader(str(DATA_DIR / "guide.pdf"), mode="page")
    show_documents("PyPDFLoader", pdf_loader.load())

    # lazy_load() yields one page at a time instead of returning a full list.
    # Process each document directly; these are pages, not text-splitter chunks.
    print("\n=== PyPDFLoader.lazy_load(): one page at a time ===")
    for page_number, document in enumerate(pdf_loader.lazy_load(), start=1):
        print(f"\nPage {page_number}")
        print("page_content:", document.page_content.strip())
        print("metadata:", document.metadata)

    # 3. PyMuPDFLoader: read the same PDF with a different parser.
    pymupdf_loader = PyMuPDFLoader(str(DATA_DIR / "guide.pdf"), mode="page")
    show_documents("PyMuPDFLoader", pymupdf_loader.load())

    # 4. CSVLoader: each row becomes a document with column names and values.
    csv_loader = CSVLoader(str(DATA_DIR / "courses.csv"), encoding="utf-8")
    show_documents("CSVLoader", csv_loader.load())

    # 5. JSONLoader: jq selects records; content_key selects their text field.
    json_loader = JSONLoader(
        file_path=str(DATA_DIR / "lessons.json"),
        jq_schema=".lessons[]",
        content_key="description",
    )
    show_documents("JSONLoader", json_loader.load())

    # 6. WebBaseLoader: fetch text from a public web page.
    web_loader = WebBaseLoader(
        web_paths=("https://example.com",),
        requests_kwargs={"timeout": 10},
        bs_get_text_kwargs={"separator": " ", "strip": True},
    )
    show_documents("WebBaseLoader", web_loader.load())

    # 7. DirectoryLoader: select text files and delegate to TextLoader.
    directory_loader = DirectoryLoader(
        str(DATA_DIR / "notes"),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    show_documents("DirectoryLoader", directory_loader.load())

    # 8. UnstructuredLoader: parse the CSV locally into a table element.
    # Compare its table metadata with CSVLoader's individual row documents.
    unstructured_loader = UnstructuredLoader(
        file_path=str(DATA_DIR / "courses.csv"),
        partition_via_api=False,
    )
    show_documents("UnstructuredLoader", unstructured_loader.load())


if __name__ == "__main__":
    main()
