## Introduction

This module provides a foundation for understanding LangChain and how it helps
developers build applications around large language models.

We will begin with the basic concepts, learn why frameworks such as LangChain
are useful, explore its main components, and see how those components fit
together in a high-level application architecture.

Each concept is supported by a small playground example. The goal is to first
understand the role of each building block, then connect the ideas into simple
workflows before moving on to advanced topics such as retrieval, tools, and
agents.

## What Is LangChain?
LangChain is an open-source framework designed to simplify building applications powered by large language models (LLMs).

If an LLM is a standalone brain or engine, LangChain acts as the vehicle or plumbing around it. It connects the model to external data sources, user memory, APIs, and tools so it can perform complex, real-world tasks.

### Core Components

The components below are ordered from a basic model call to more advanced
retrieval-augmented and agent-based applications:

- `Models`: Provide a consistent interface for connecting to different LLMs, such as GPT, Claude, or Gemini.
- `Prompt Templates`: Create reusable prompts by combining fixed instructions with dynamic input values.
- `Output Parsers`: Convert model responses into predictable formats such as strings, JSON, or typed objects.
- `Chains`: Connect prompts, models, parsers, and other steps into a repeatable workflow.
- `Document Loaders`: Load content from sources such as PDFs, web pages, files, databases, and YouTube transcripts.
- `Text Splitters`: Break large documents into smaller chunks that are easier for models to process and retrieve.
- `Embeddings`: Convert text into numerical representations that capture semantic meaning.
- `Vector Stores`: Store and index embeddings so semantically similar content can be searched efficiently.
- `Retrievers`: Find and return relevant documents or document chunks for a question or prompt.
- `Tools`: Give models and agents access to external functions, APIs, databases, and services.
- `Agents`: Let an LLM decide which tools or APIs to use to complete a task.
- `Memory`: Preserve relevant conversation history or application state across interactions.
- `Callbacks and Observability`: Monitor model calls, token usage, latency, errors, and application traces.

## Why Do We Need LangChain?

An LLM can generate a response from a prompt, but real applications usually
need more than a single model call. They may need to load company documents,
retrieve relevant information, call external tools, preserve conversation
state, and format the final response consistently.

LangChain provides reusable components for connecting these steps into an
application workflow. It also offers common interfaces for models and tools,
which can make it easier to change providers or replace individual parts of a
workflow.

For example, a question-answering application might follow this flow:

```text
user question
-> retrieve relevant documents
-> build a prompt with the retrieved context
-> call the language model
-> parse and return the answer
```

LangChain is not required for every application. For a single model call, a
provider's SDK may be simpler. LangChain becomes more useful as the workflow
grows and needs composition, integrations, state, tools, or observability.

## High-Level Application Architecture

An application built with LangChain usually connects the user, application
logic, LangChain components, a model provider, and optional external systems.
LangChain coordinates the workflow, while the model provider generates the
language-based response.

```text
user input
  |
  v
application
  |
  v
LangChain workflow
  |
  +--> prompt template
  +--> model call ------------------> OpenAI or another model provider
  +--> output parser
  |
  +--> retriever --> vector store or other data source
  |
  +--> tools ------> APIs, databases, or application services
  |
  v
application response
```

For a simple application, the workflow may only contain a prompt and a model:

```text
user input -> prompt template -> model -> application response
```

For a retrieval-augmented application, relevant information is fetched before
the model is called:

```text
user question
-> retriever
-> relevant context
-> prompt template
-> model
-> final answer
```

This architecture makes each step replaceable. For example, an application can
change its model provider, retriever, prompt, or output parser without
rewriting the entire workflow.

## Core Building Blocks

LangChain applications are built by composing small steps called **runnables**.
A runnable accepts an input, performs one operation, and returns an output. A
prompt template, model, output parser, retriever, and many other components can
be used as runnables.

Several runnables can be connected into a **chain**. LangChain Expression
Language (LCEL) commonly uses the pipe operator (`|`) to express this flow:

```python
chain = prompt | model | output_parser
```

The data moves from left to right. The prompt creates the model input, the
model generates a response, and the output parser converts that response into
a useful application value.

The central idea is composition:

```text
input -> prompt -> model -> parser -> application result
```

## Benefits of LangChain

LangChain provides several practical benefits when building applications with
language models:

- **Composition**: Combine prompts, models, parsers, retrievers, and tools into
  clear, reusable workflows.
- **Provider flexibility**: Use a common interface so changing model providers
  requires fewer changes to application code.
- **Integration support**: Connect language models to documents, vector stores,
  APIs, databases, and other external systems.
- **Reusability**: Turn common prompts and workflow steps into components that
  can be shared across features.
- **Structured output**: Convert unpredictable model responses into strings,
  JSON, or typed application objects.
- **Streaming and execution modes**: Invoke workflows one input at a time,
  stream responses, or process multiple inputs in batches.
- **Observability**: Trace workflow steps and inspect latency, token usage, and
  errors during development and production.
- **Easier evolution**: Replace one part of a workflow without rewriting the
  entire application.

These benefits are most valuable when an application has multiple steps or
integrations. For a simple one-off model call, the provider's SDK may be a
better choice because it introduces less abstraction.

## What Can You Build with LangChain?

LangChain can support applications that combine language models with prompts,
data sources, tools, and application logic. Common examples include:

- **Chatbots and assistants**: Build conversational applications that maintain
  context and respond using a model.
- **Document question-answering**: Let users ask questions about PDFs, web
  pages, internal documents, or other knowledge sources.
- **Retrieval-augmented generation (RAG)**: Retrieve relevant information and
  provide it to a model as context before generating an answer.
- **Summarization and extraction**: Summarize long content or extract
  structured information such as names, dates, and classifications.
- **Content generation**: Create drafts, reports, emails, descriptions, and
  other content from structured or unstructured input.
- **Research assistants**: Search multiple sources, compare information, and
  produce a grounded response.
- **Tool-using applications**: Connect models to APIs, databases, calculators,
  search systems, and other external functions.
- **Workflow automation**: Orchestrate multi-step processes such as ticket
  classification, data enrichment, and report generation.
- **Agents**: Build applications where a model chooses which tools to use and
  in what order to complete a task.

The best architecture depends on the problem. Some applications need only a
prompt and a model, while others may combine retrieval, tools, memory, agents,
and structured output.

## Alternatives to LangChain

LangChain is one option for building LLM applications. The best choice depends
on the application's complexity, the team's preferences, and the integrations
that are required.

- **Model provider SDKs**: OpenAI, Anthropic, Google, and other providers offer
  their own SDKs. These are often the simplest choice for direct model calls
  and provider-specific features.
- **LlamaIndex**: Focuses strongly on connecting LLMs to external data,
  document indexing, and retrieval-augmented generation workflows.
- **Microsoft Semantic Kernel**: Provides orchestration, plugins, memory, and
  planning features with strong support for Microsoft and enterprise
  ecosystems.
- **Haystack**: Offers components and pipelines for search, retrieval,
  question-answering, and production NLP applications.
- **DSPy**: Focuses on programming and optimizing language model pipelines
  through declarative modules and evaluation-driven optimization.
- **Custom application code**: For a small application, direct SDK calls and
  ordinary Python functions may be clearer than adding a framework.

LangChain is a good fit when an application needs composable workflows,
multiple integrations, provider flexibility, or a common interface across
models and tools. It may be unnecessary when the application only makes a
small number of simple model calls.

## Playground

The accompanying [playground](playground.py) demonstrates the smallest useful
chain: a prompt connected to an OpenAI chat model.