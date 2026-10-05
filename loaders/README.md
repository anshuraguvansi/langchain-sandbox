# LangChain document loaders

## What are loaders?

Document loaders read data from sources such as files, websites, and databases
and convert it into LangChain `Document` objects. This gives downstream steps a
consistent format to work with, regardless of where the data came from.

Each document contains:

- **`page_content`**: the loaded text.
- **`metadata`**: information about the text, such as its source, page number, or
  title. The available fields depend on the loader.

Loaders are often the first step in a retrieval-augmented generation (RAG)
pipeline. After loading, documents can be split into smaller chunks, embedded,
and stored for retrieval. Loading itself does not create embeddings or train a
model.

## Commonly used loaders

| Loader | What it loads |
| --- | --- |
| `TextLoader` | Plain text files. |
| `PyPDFLoader` | PDF text and metadata using `pypdf`, with plain or layout text extraction. |
| `PyMuPDFLoader` | PDF text and metadata using `PyMuPDF`, with optional table extraction as Markdown, HTML, or CSV. |
| `CSVLoader` | CSV data, typically producing one document per row. |
| `JSONLoader` | Content selected from JSON or JSON Lines using a jq schema. |
| `WebBaseLoader` | Text from HTML web pages. |
| `DirectoryLoader` | Files in a directory, using a chosen loader to read each file. |
| `UnstructuredLoader` | Content from multiple file formats through the Unstructured integration. |

Choose a loader based on the source format and the metadata you need. Some
loaders require additional packages or service credentials.

