# LangChain Models

Models process information and produce either vectors or generated responses.
LangChain provides common interfaces so applications can use models from
different providers in composable workflows.

This module focuses on model and message concepts. Code examples are available
in the playground.

## Main Model Types

### Embedding models

Embedding models convert text into numerical vectors that represent semantic
meaning. They are used for semantic search, document indexing, similarity
comparison, RAG, clustering, and classification features.

Embedding models do not generate answers. They help an application find the
most relevant information:

```text
text -> embedding model -> vector
```

The document and query embedding models should be compatible. Vector dimensions,
distance metric, chunking strategy, and metadata all affect retrieval quality.

### Language models (LLMs)

Traditional language models accept a text prompt and return generated text:

```text
string prompt -> language model -> string completion
```

They are useful for completion, summarization, extraction, classification, and
other text-generation tasks.

### Chat models

Chat models accept structured messages and return an assistant message. They
are commonly used for conversations, tool calling, structured output, and
multimodal applications:

```text
messages -> chat model -> AIMessage
```

Chat models are still language models, but the message format provides more
structure than a single text prompt.

### Multimodal models

Some chat models can process content beyond text, such as images, audio, and
files. Support varies by provider and model, so applications should verify the
supported input and output modalities.

## LangChain Model Interfaces

| Interface | Input | Output | Purpose |
| --- | --- | --- | --- |
| `Embeddings` | Text | Vectors | Search and similarity |
| `BaseLLM` | Text prompt | Generated text | Completion workflows |
| `BaseChatModel` | Messages | `AIMessage` | Conversations and tool use |
| `BaseLanguageModel` | Language-model input | Model-specific | Shared base abstraction |

Provider integrations implement these interfaces. This gives applications a
consistent way to switch providers or compose model calls with other
LangChain components.

## The Runnable Model Lifecycle

Models are runnables, so they support common execution methods:

- `invoke`: Run one input and return the complete result.
- `stream`: Receive output incrementally.
- `batch`: Process multiple inputs.
- `ainvoke`, `astream`, and `abatch`: Async equivalents.

The same runnable interface is used by many LangChain components. A model can
therefore be composed with other runnable steps in an application workflow:

```text
input -> model -> application result
```

## Messages

Messages are the basic input and output units for chat models:

- `SystemMessage`: Application instructions and behavior.
- `HumanMessage`: User input.
- `AIMessage`: Model response, tool calls, and usage information.
- `ToolMessage`: The result returned by an external tool.

Chat models do not automatically remember previous requests. Applications must
send relevant message history with each call or store it in application state.
As history grows, it may need to be trimmed or summarized to stay within the
model's context window.

## Embedding and Generation Roles

Applications commonly use both model families for knowledge-based answers:

```text
source text -> embedding model -> vector representation

user question -> chat model -> generated answer
```

The embedding model represents meaning for comparison and search. The language
or chat model generates text. Keeping these responsibilities separate makes it
easier to choose the right model and diagnose quality problems.

## Configuration and Model Selection

Common configuration includes model identifier, temperature or other sampling
controls, maximum output tokens, timeout, retry behavior, and provider-specific
features.

Choose a model based on the task:

| Task | Model choice | Evaluate |
| --- | --- | --- |
| Semantic search | Embedding model | Relevance, recall, dimensions, cost |
| Extraction or classification | Small LLM or chat model | Accuracy and schema adherence |
| Conversation | Chat model | Helpfulness and context handling |
| RAG | Embedding model plus chat model | Retrieval and groundedness |

Evaluate candidate models with representative inputs. Measure quality,
groundedness, latency, token usage, rate-limit behavior, and total cost.

## Context, Tokens, and Reliability

Models process tokens and have a finite context window. A request's context may
include system instructions, conversation history, retrieved documents, tool
results, the current question, and the requested answer.

Too much context can increase cost, reduce answer quality, or exceed the model's
limit. Keep retrieved content relevant and reserve enough space for the output.

Temperature affects variation; it does not guarantee factuality. Reliability
also depends on prompt design, model quality, retrieved context, tool
validation, output validation, retries, and evaluation.

## Common Mistakes

- Using an embedding model to generate an answer.
- Treating a chat model's `AIMessage` as a plain string without extracting its
  content.
- Assuming chat history is automatically remembered.
- Changing embedding models without rebuilding a vector index.
- Sending all available documents instead of relevant context.
- Ignoring provider rate limits, timeouts, and transient failures.

### Playground
- See the [playground](playground.py) for runnable examples and experiments.
- Use this command to run example `uv run --env-file .env models/playground.py`
