"""Structured-output examples for direct chat-model calls.

Run with:

    uv run --env-file .env python structured-output/playground.py

The examples use both strategies supported by LangChain model integrations:
provider-native structured output and tool-calling structured output.
"""

from dataclasses import dataclass
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Person(BaseModel):
    """Information extracted about one person."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age", ge=0)
    occupation: str = Field(description="The person's occupation")


@dataclass
class PersonDataclass:
    """Information extracted about one person."""

    name: str
    age: int
    occupation: str


class PersonTypedDict(TypedDict):
    """Information extracted about one person."""

    name: str
    age: int
    occupation: str


PERSON_JSON_SCHEMA = {
    "title": "Person",
    "description": "Information extracted about one person.",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The person's full name"},
        "age": {"type": "integer", "description": "The person's age"},
        "occupation": {
            "type": "string",
            "description": "The person's occupation",
        },
    },
    "required": ["name", "age", "occupation"],
    "additionalProperties": False,
}


SCHEMAS = [
    ("Pydantic model", Person),
    ("Dataclass", PersonDataclass),
    ("TypedDict", PersonTypedDict),
    ("JSON Schema", PERSON_JSON_SCHEMA),
]


def run_schema_examples(
    method: Literal["json_schema", "function_calling", "json_mode"],
) -> None:
    """Run every schema with one structured-output strategy."""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    strategy_name = {
        "json_schema": "Provider strategy (native JSON Schema)",
        "function_calling": "Tool strategy (function calling)",
        "json_mode": "JSON mode (alternative structured-output strategy)",
    }[method]

    print(f"\n=== {strategy_name} ===")

    for schema_name, schema in SCHEMAS:
        structured_model = model.with_structured_output(schema, method=method)
        result = structured_model.invoke(
            "Extract the person from this sentence: "
            "Ada Lovelace was a 36-year-old mathematician."
        )
        print(f"{schema_name}: {result}")


def main() -> None:
    # Provider strategy: the model provider enforces the response schema.
    run_schema_examples(method="json_schema")

    # Tool strategy: LangChain represents the schema as a tool call.
    run_schema_examples(method="function_calling")

    # JSON mode: an alternative structured-output strategy.
    run_schema_examples(method="json_mode")


if __name__ == "__main__":
    main()
