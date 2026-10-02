"""Small, model-free examples of commonly used LangChain output parsers.

Run with:

    uv run --env-file .env python output-parser/playground.py

The examples parse fixed model-like responses, so no API key is required.
"""

from enum import Enum

from langchain_classic.output_parsers import (
    BooleanOutputParser,
    DatetimeOutputParser,
    EnumOutputParser,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    PydanticOutputParser,
    StrOutputParser,
    XMLOutputParser,
)
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    """Allowed sentiment values."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Review(BaseModel):
    """Validated structured output for a review."""

    sentiment: Sentiment
    score: int = Field(ge=1, le=5)


class KeyValueParser(BaseOutputParser[dict[str, str]]):
    """Parse one `key: value` pair per line."""

    def parse(self, text: str) -> dict[str, str]:
        result: dict[str, str] = {}

        for line in text.strip().splitlines():
            if ":" not in line:
                raise OutputParserException(f"Expected 'key: value', got: {line!r}")
            key, value = line.split(":", maxsplit=1)
            if not key.strip() or not value.strip():
                raise OutputParserException(f"Invalid key-value line: {line!r}")
            result[key.strip()] = value.strip()

        if not result:
            raise OutputParserException("Expected at least one key-value pair")
        return result


def main() -> None:
    # Parsers are runnables, so they can be called with invoke().
    print("String via invoke:", StrOutputParser().invoke("Hello from LangChain."))

    # A small transformation can also be composed as a runnable parser step.
    clean_text = RunnableLambda(lambda text: text.strip().upper())
    print("Lambda parser:", clean_text.invoke("  hello from lambda  "))

    print(
        "JSON:",
        JsonOutputParser().parse('{"name": "Ada", "age": 36}'),
    )

    review_parser = PydanticOutputParser(pydantic_object=Review)
    print(
        "Pydantic:",
        review_parser.parse('{"sentiment": "positive", "score": 5}'),
    )

    print(
        "Comma-separated list:",
        CommaSeparatedListOutputParser().parse("python, langchain, ai"),
    )
    print(
        "Numbered list:",
        NumberedListOutputParser().parse("1. Prompt\n2. Model\n3. Parser"),
    )
    print(
        "Markdown list:",
        MarkdownListOutputParser().parse("- Prompt\n- Model\n- Parser"),
    )
    print("Boolean:", BooleanOutputParser().parse("YES"))
    print(
        "Datetime:",
        DatetimeOutputParser().parse("2026-10-02T12:30:00.000000Z"),
    )
    print("Enum:", EnumOutputParser(enum=Sentiment).parse("neutral"))
    print(
        "XML:",
        XMLOutputParser().parse("<person><name>Ada</name></person>"),
    )
    print(
        "Custom:",
        KeyValueParser().parse("name: Ada Lovelace\nrole: mathematician"),
    )


if __name__ == "__main__":
    main()
