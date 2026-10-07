# LangChain text splitters

## What are splitters?

Text splitters divide loaded content into smaller chunks so it can be embedded,
retrieved, and passed to a model within its input limits. In a typical
retrieval-augmented generation (RAG) pipeline, splitting comes after loading:

**Load documents → Split into chunks → Embed → Store → Retrieve**

When splitting LangChain `Document` objects with `split_documents()`, the chunks
retain the source metadata, such as a filename or page number. This helps trace
retrieved content back to its origin.

## Commonly used splitters

| Splitter | How it splits and when to use it |
| --- | --- |
| `RecursiveCharacterTextSplitter` | Tries paragraph breaks, line breaks, spaces, and finally individual characters. A good starting point for general text and text extracted from PDFs. |
| `CharacterTextSplitter` | Splits on a chosen separator, such as a blank line, then combines pieces into chunks. Useful for consistently formatted text; an individual piece can exceed `chunk_size`. |
| `TokenTextSplitter` | Splits by token count using `tiktoken`. Useful for token budgets; select a tokenizer that matches the target model. |
| `MarkdownHeaderTextSplitter` | Splits Markdown by configured heading levels and records headings in metadata. Useful for documentation and knowledge bases. |
| `HTMLHeaderTextSplitter` | Groups HTML content by headings and records heading metadata. Useful when web page sections provide meaningful context. |
| `RecursiveJsonSplitter` | Recursively splits JSON while preserving nested structure where possible. Useful for large structured payloads; long string values may exceed the requested size. |
| `RecursiveCharacterTextSplitter.from_language(...)` | Uses language-specific separators for source code, such as class and function boundaries. Useful for code retrieval; this is a factory method, not a separate splitter class. |
| `NLTKTextSplitter` / `SpacyTextSplitter` | Use sentence segmentation to group text into chunks. Useful when sentence boundaries matter; require additional NLP packages and potentially language resources. |

These are practical options to learn, rather than a measured popularity ranking.
See the official [splitter overview](https://docs.langchain.com/oss/python/integrations/splitters/index)
and guides for [characters](https://docs.langchain.com/oss/python/integrations/splitters/character_text_splitter),
[tokens and sentences](https://docs.langchain.com/oss/python/integrations/splitters/split_by_token),
[JSON](https://docs.langchain.com/oss/python/integrations/splitters/recursive_json_splitter),
and [code](https://docs.langchain.com/oss/python/integrations/splitters/code_splitter).

## Key settings

For size-based text splitters:

- **`chunk_size`**: the desired chunk limit. Character splitters measure
  characters by default; token splitters measure tokens. These are not equivalent.
- **`chunk_overlap`**: the target amount of repeated content between neighboring
  chunks to preserve context. Keep it smaller than `chunk_size`; actual overlap
  depends on split boundaries.
- **Separators**: the boundaries used to divide text. Recursive splitting tries
  several in order, falling back to finer boundaries when needed.

Start with `RecursiveCharacterTextSplitter` and inspect the resulting chunks.
Tune size and overlap against retrieval quality and the embedding model's input
limit. The official [recursive splitting guide](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
explains these settings.

Header splitters group content by structure and do not enforce a chunk size.
Follow them with a recursive splitter when sections are too large. For token
limits with Unicode text, prefer `RecursiveCharacterTextSplitter.from_tiktoken_encoder()`
over direct token slicing, which can split a character's tokens across chunks.
See the [token splitting guide](https://docs.langchain.com/oss/python/integrations/splitters/split_by_token).

## Examples: how each splitter behaves

Sizes count
characters unless tokens are explicitly mentioned. Overlap is zero unless
specified otherwise.

### 1. RecursiveCharacterTextSplitter

Imagine two paragraphs:

> Red fox jumps over the log.
>
> Blue bird sings.

With `chunk_size=20`, it first tries the paragraph break. The first paragraph
has 26 characters, so it needs further splitting. There are no line breaks
inside it, so the splitter tries spaces and groups words that fit within 20
characters. The second paragraph already fits.

The resulting chunks are:

- **Red fox jumps over** — 18 characters.
- **the log.** — 8 characters.
- **Blue bird sings.** — 16 characters.

The default order is paragraph breaks → line breaks → spaces → individual
characters. Only oversized pieces move to the next level. Pieces that fit can
be combined into chunks; a complete tree is not built first.

With the default separators, a 200-character paragraph and a size limit of 100
will be split further until every chunk fits. Chunks may be smaller than 100.
The final character-level fallback makes this possible; removing it can allow
oversized chunks. Overlap depends on the available boundaries.

### 2. CharacterTextSplitter

This splitter works in two steps:

1. **Split at the chosen separator** to produce pieces.
2. **Combine consecutive pieces while the total fits within `chunk_size`**,
   counting separators too. If adding the next piece would exceed the target,
   finish the current chunk and start another.

For example, suppose paragraph breaks produce pieces of **30, 40, and 50
characters**, with `chunk_size=100` and no overlap. A paragraph break is
`\n\n`, which counts as two characters.

- Combine the first two: 30 + 2 + 40 = **72 characters**.
- Adding the third would make 72 + 2 + 50 = **124 characters**, exceeding 100.
- The final chunks are **72 and 50 characters**.

It does not keep adding pieces to reach or exceed 100. Smaller chunks are
expected when the next piece will not fit.

Imagine three paragraphs containing **AAAA**, **BBBBB**, and **CCCCCC**,
separated by blank lines. Choose the blank line as the separator and set
`chunk_size=12`.

It first produces pieces of 4, 5, and 6 characters. It then combines the first
two into one chunk: 4 + 2 characters for the blank line + 5 = **11 characters**.
Adding the third would exceed 12, so that piece becomes a separate chunk.

The result is:

- Chunk 1: **AAAA**, a blank line (`\n\n`), then **BBBBB** — 11 characters.
- Chunk 2: **CCCCCC** — 6 characters.

An individual piece can already exceed the target. If the separator is a
paragraph break, a 200-character paragraph with no internal paragraph break
remains a 200-character chunk even with `chunk_size=100`. The combining step
cannot make that piece smaller. Use `RecursiveCharacterTextSplitter` when
oversized pieces need further splitting.

### 3. TokenTextSplitter

Imagine the tokenizer turns some text into **12 tokens**, labeled T1 through
T12. Set `chunk_size=5` tokens and `chunk_overlap=1` token.

The windows are:

- Chunk 1: **T1–T5**.
- Chunk 2: **T5–T9**.
- Chunk 3: **T9–T12**.

T5 and T9 are repeated to provide overlap. Each window contains at most 5
original tokens, but its character count varies. A token may be a word, part
of a word, punctuation, or whitespace, depending on the tokenizer.

This splitter follows token positions, so windows may cross sentence
boundaries. Direct token slicing can also split a Unicode character encoded
as multiple tokens.

### 4. Recursive splitting with a token budget

Imagine a paragraph that measures **18 tokens**, with a limit of **10 tokens**.
The splitter tries paragraph breaks first. Since the paragraph is too large,
it tries line breaks and then spaces to make smaller pieces.

It combines those pieces while counting tokens rather than characters. For
example, a boundary-respecting split might yield chunks of **9 and 9 tokens**;
the actual counts depend on the text and tokenizer.

A paragraph already within the budget does not need deeper splitting. This
uses the same recursive strategy as the character-based version while
preserving Unicode character boundaries. Overlap is a target based on the
pieces available to repeat.

### 5. MarkdownHeaderTextSplitter

Imagine a Markdown document with this heading structure:

- Title: **Guide**.
- Section: **Loading**, containing “Read files.”
- Section: **Splitting**, containing “Create chunks.”

When configured to split by the title and section heading levels, it produces:

| Content | Heading metadata |
| --- | --- |
| Read files. | Title: Guide; Section: Loading |
| Create chunks. | Title: Guide; Section: Splitting |

Headings define the sections and are removed from content by default, while
remaining available in metadata. There is no `chunk_size` setting here.

If the Loading section contains 2,000 characters, it can remain one section.
Pass that section through a recursive splitter afterward to create smaller
chunks that retain the same heading metadata.

### 6. HTMLHeaderTextSplitter

Imagine a web page headed **Guide**, with two subheadings:

- **Loading**, followed by “Read files.”
- **Splitting**, followed by “Create chunks.”

When configured to use these heading levels, the splitter associates “Read
files.” with **Guide → Loading** and “Create chunks.” with **Guide → Splitting**.
Those heading paths become metadata. The output can also contain documents
for the headings themselves.

It groups content using the page's HTML heading structure. A long section can
still be a long chunk because this splitter does not enforce a character
budget. A recursive splitter can divide those sections afterward.

### 7. RecursiveJsonSplitter

Imagine a JSON object called **guide** containing three entries: **loading**,
**splitting**, and **retrieval**. Together, they exceed a 100-character target
when serialized, but each entry fits individually with its surrounding keys.

The splitter can divide the object into three smaller objects:

- Chunk 1 contains **guide → loading** and its value.
- Chunk 2 contains **guide → splitting** and its value.
- Chunk 3 contains **guide → retrieval** and its value.

Each chunk retains the parent **guide** key, so the original nesting remains
clear. Size counts the serialized JSON, including keys, quotes, and punctuation.
Smaller entries can share a chunk when they fit; `min_chunk_size` influences
when another chunk is started.

If one value is a 500-character string, it is not sliced into smaller strings.
That chunk can exceed the target. Splitting happens through the JSON structure.

### 8. Language-aware recursive splitting

Imagine a Python file with a **load** function of 30 characters and a **split**
function of 32 characters. With `chunk_size=40`, the two functions cannot fit
in one chunk.

Python-specific separators favor function boundaries, so each function can
become its own chunk. If a function is 200 characters, the splitter moves to
finer boundaries such as lines and spaces until the pieces fit.

This is recursive splitting with separators selected for the programming
language. It uses text patterns rather than a syntax tree, so a large function
can be divided into fragments that are not valid standalone code.

### 9. NLTKTextSplitter

Imagine the text:

> Load files. Split text. Retrieve chunks.

NLTK first identifies three sentences. With `chunk_size=30`, the splitter
combines the first two, joining them with a blank line by default:

- Chunk 1: **Load files.**, a blank line, then **Split text.** — 23 characters.
- Chunk 2: **Retrieve chunks.** — 16 characters.

All three together would exceed 30. The boundaries come from sentence
segmentation; `chunk_size` controls how sentences are combined. A single
200-character sentence remains whole even if the target is 100.

### 10. SpacyTextSplitter

For the same text, spaCy's rule-based **sentencizer** also identifies “Load
files.”, “Split text.”, and “Retrieve chunks.” as separate sentences.

With `chunk_size=30`, it combines the first two into a 23-character chunk and
leaves the third as a 16-character chunk, just like the NLTK example.

The sentence detector differs: the playground uses spaCy's rule-based
sentencizer, which needs no downloaded language model. Other spaCy pipelines
can use a language model to identify boundaries. It still groups whole
sentences, so an oversized sentence can exceed the size target.

### 11. Custom FixedWidthTextSplitter

Imagine the input **ABCDEFGHIJKL**, with `chunk_size=5` and `chunk_overlap=2`.

The resulting chunks are:

- **ABCDE** — positions 0–4.
- **DEFGH** — positions 3–7.
- **GHIJK** — positions 6–10.
- **JKL** — positions 9–11.

Each window starts 3 characters after the previous one: size 5 minus overlap
2. Neighboring chunks repeat exactly 2 characters. Every chunk except possibly
the last has exactly 5 characters.

This custom splitter follows character positions and can cut through words.
Its implementation is in [playground.py](playground.py). It also supports
splitting documents while retaining their source metadata.

## Run the playground

[playground.py](playground.py) demonstrates every splitter above, recursive
splitting with a token budget, and a custom `FixedWidthTextSplitter` subclass.
The custom splitter implements `split_text()` using exact character windows
with overlap; inherited `split_documents()` preserves source metadata. Unlike
recursive splitting, it can cut through words.

Run from the repository root:

```bash
uv sync --extra splitters
uv run --extra splitters python -m nltk.downloader -d .nltk_data punkt_tab
uv run --extra splitters python splitters/playground.py
```

No API key is needed. Initial setup needs internet access for packages, NLTK
sentence data, and the first token-encoding download. Downloaded data is cached
in ignored workspace folders. If NLTK data is missing, its example prints the
setup command and the remaining examples continue. The spaCy example uses the
rule-based `sentencizer`, so no spaCy language model download is needed.

Output shows each chunk's content and character count, token counts for token
examples, and metadata for `Document` outputs. Compare recursive and fixed-width
boundaries, then change `chunk_size` and `chunk_overlap` to see their effects.
