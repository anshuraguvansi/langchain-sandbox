# Structured output

Structured output makes an LLM return data in a predictable schema instead of
only free-form text. LangChain validates the result so the application can use
the data directly.

It is useful for:

- Data extraction: turning unstructured text into typed fields.
- API building: returning consistent, validated response objects.
- Agents: giving an agent a predictable final result for downstream steps.

## The core idea

You define the shape of the response, including field names, types,
descriptions, allowed values, and validation rules. The schema is the contract
between the model and your application.

Common schema types are:

- Pydantic models: typed, validated results returned as a Pydantic instance.
- Dataclasses and TypedDicts: typed definitions returned as dictionaries.
- JSON Schema: useful when the schema is defined outside Python.

## Schema type comparison

| Schema type | What it provides | Returned value | Use it when |
| --- | --- | --- | --- |
| Pydantic model | Type hints, field descriptions, and runtime validation | Pydantic instance | You want the strongest Python validation and an easy-to-use model object. |
| Dataclass | Lightweight Python data structure with type annotations | Dictionary | You want simple Python data with minimal validation or framework behavior. |
| TypedDict | Type hints for dictionary keys and values | Dictionary | You want dictionary-shaped data and static type checking. |
| JSON Schema | Language-independent schema defined as a dictionary | Dictionary | Your schema comes from an API, configuration, or another non-Python system. |

In general, start with a Pydantic model when building a Python application.
Choose a dataclass or TypedDict for lightweight internal data, and JSON Schema
when the schema must be shared across languages or systems.

## Choosing a strategy

LangChain supports two ways to produce structured output:

- Provider strategy: the model provider enforces the schema natively. This is
  generally the most reliable option when the selected model supports it.
- Tool strategy: LangChain represents the schema as a tool call. Use this for
  models without native structured-output support but with tool-calling
  support.

When a schema type is passed directly to an agent, LangChain automatically
chooses the provider strategy when supported and otherwise falls back to the
tool strategy. An explicit strategy is useful when you need to control that
choice. JSON Schema dictionaries must be wrapped in an explicit strategy.

For direct model calls, the model's structured-output interface uses the same
schema concepts and provider/tool options. The exact method depends on the
model integration.

## What to remember

- Request structured output through the model or agent's structured-output
  interface. Agents use `response_format`.
- Direct model calls return the structured value through the model response;
  agents expose it in `structured_response`.
- A schema can include constraints, not only field types; clear field
  descriptions improve results.
- Tool strategy can retry when the model returns invalid data. Error handling
  is configurable: retry by default, handle selected errors, customize the
  retry message, or let errors propagate.
- Tool strategy also handles the case where a model returns multiple
  structured outputs when only one is expected.
- If the model must use tools as well as return structured data, it must
  support using both capabilities at the same time.

## Model output and agent output

Structured output can be produced directly by a chat model or by an agent
created with `create_agent`. The agent workflow adds agent state and exposes
the final structured value in `structured_response`; the underlying idea and
schema contract are the same.
