# LangChain Sandbox

A step-by-step learning workspace for LangChain. Each module combines a short
theory guide with a small playground that can be run and modified locally.

Official LangChain documentation: [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)

## Learning Path

| Module | Topic | Status |
| --- | --- | --- |
| [01 - Introduction](introduction/README.md) | What LangChain is, the core building blocks, and a first runnable example | Completed |
| [02 - Models](models/README.md) | Embeddings, LLMs, chat models, multimodal models, and model composition | Completed |
| [03 - Prompts](prompts/README.md) | Prompt templates, chat prompts, variables, composition, and few-shot prompting | Completed |
| [04 - Structured output](structured-output/README.md) | Schemas, provider and tool strategies, validation, data extraction, APIs, and agent results | Completed |
| [05 - Output parsers](output-parser/README.md) | Convert model responses into text, JSON, lists, typed values, and custom application formats | Completed |

More modules will be added here as the learning path grows.

## Repository Structure

Each module combines a theory guide with runnable examples:

```text
<module>/
├── README.md          # theory, concepts, and exercises
└── playground.py      # runnable examples for the module
```

Some modules organize multiple playgrounds in a subdirectory instead of using
one file directly.

## Prerequisites

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key

Install `uv` if it is not already available, then sync the project:

```bash
uv sync
```

`uv` creates and manages the project environment from `pyproject.toml` and
`uv.lock`. Add dependencies with `uv add`; do not install them globally.

Create a `.env` file in the repository root and add your OpenAI API key:

```dotenv
OPENAI_API_KEY=""
```

Replace the empty value with your key. The `.env` file is ignored by Git and
must never be committed or shared.

Run an example with the environment file loaded:

```bash
uv run --env-file .env python introduction/playground.py
```

For example, the prompts playgrounds are in `prompts/playground/`, the
structured-output examples are in `structured-output/playground.py`, and the
output-parser examples are in `output-parser/playground.py`.
Module-specific setup and commands live in each module's README.

## How to Use This Repository

1. Read the module theory in its `README.md`.
2. Run the playground and inspect the output.
3. Change one thing at a time and rerun it.
4. Complete the exercises before moving to the next module.
