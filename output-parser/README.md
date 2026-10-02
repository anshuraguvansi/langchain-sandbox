# LangChain output parsers

Output parsers turn a model response into a value your application can use.
They are usually the last step in an LCEL chain:

```text
prompt -> model -> output parser -> application value
```

## Core parsers

Import the commonly used parsers from `langchain_core.output_parsers`:

| Parser | Returns | Use it for |
| --- | --- | --- |
| `StrOutputParser` | `str` | Plain text; also extracts content from an `AIMessage`. |
| `JsonOutputParser` | `dict` or `list` | JSON when you need a flexible shape or streaming JSON. |
| `PydanticOutputParser` | Pydantic model | JSON validated against a Python model. |
| `SimpleJsonOutputParser` | `dict` or `list` | Alias for the simple JSON parser. |
| `CommaSeparatedListOutputParser` | `list[str]` | Values such as `red, green, blue`. |
| `ListOutputParser` | `list[str]` | One item per line. |
| `NumberedListOutputParser` | `list[str]` | Numbered output such as `1. first`. |
| `MarkdownListOutputParser` | `list[str]` | Markdown bullets such as `- first`. |
| `XMLOutputParser` | parsed XML data | XML responses. |

`ListOutputParser` is an abstract base class; use a concrete list parser such
as `NumberedListOutputParser` or `MarkdownListOutputParser`.

For JSON or Pydantic parsing, include the parser's formatting instructions in
the prompt. Prefer a model's native `with_structured_output(...)` when the
provider supports it; it usually gives a stronger schema contract than asking
the model to print JSON manually.

## Custom parsers

Use a custom parser when the built-in parsers do not match your output format.
Subclass `BaseOutputParser`, implement `parse()`, and optionally provide
`get_format_instructions()`. The parser can then be used like any other
runnable in a chain.

Keep custom parsers small and deterministic: validate the complete response,
raise a clear parse error for invalid output, and keep business rules outside
the parser. See `playground.py` for an implementation example.

## Repair and retry parsers

These parsers wrap another parser and react to invalid model output:

- `OutputFixingParser`: asks an LLM to fix malformed output.
- `RetryOutputParser`: retries the original prompt with parser-error context.
- `RetryWithErrorOutputParser`: retries while explicitly including the error.

They can improve reliability, but validation, clear format instructions, and
provider-native structured output are usually better first choices.

## Other parser families

- **Transform parsers** such as `BaseTransformOutputParser` and
  `BaseCumulativeTransformOutputParser` parse streamed chunks.
- **Combining parsers** such as `CombiningOutputParser` combine results from
  multiple parsers.
- **Agent parsers** interpret tool calls and final answers, for example
  `OpenAIToolsAgentOutputParser`, `ToolsAgentOutputParser`, and
  `XMLAgentOutputParser`. Use these for agent internals, not ordinary chains.
- **Structured text parsers** such as `StructuredOutputParser` and
  `ResponseSchema` are older JSON-format helpers. For new code, use
  `JsonOutputParser`, `PydanticOutputParser`, or native structured output.
- **Classic parsers** such as `BooleanOutputParser` (expects `YES` or `NO`),
  `DatetimeOutputParser`, `EnumOutputParser`, and `RegexParser` are available from
  `langchain_classic.output_parsers` when you need them.

## Choosing quickly

```text
Need plain text?                  -> StrOutputParser
Need flexible JSON?               -> JsonOutputParser
Need validated typed data?        -> PydanticOutputParser or native structured output
Need a small list / boolean?      -> the matching scalar/list parser
Need streaming structured data?   -> JsonOutputParser or a transform parser
Need an agent tool/action result? -> an agent output parser
```

Parsers are not a safety boundary: validate values, handle parse errors, and
apply authorization checks in application code.

Further reading: [LangChain output parser reference](https://python.langchain.com/api_reference/core/output_parsers.html)
and [structured output](https://docs.langchain.com/oss/python/langchain/structured-output).
