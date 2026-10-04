"""Examples of common LangChain chain patterns.

Run with:

    uv run --env-file .env python chains/playground.py

Set ``GOOGLE_API_KEY`` in ``.env`` before running. Each function demonstrates
one composition pattern and uses the Gemini chat model.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

parser = StrOutputParser()
model: ChatGoogleGenerativeAI | None = None


def get_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini model only when an example is executed."""

    global model
    if model is None:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    return model


def sequence_chain() -> None:
    """Run steps in order: prompt -> model -> parser."""

    chat_model = get_model()
    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    chain = prompt | chat_model | parser
    print("Sequence:", chain.invoke({"topic": "LangChain chains"}))


def parallel_chain() -> None:
    """Run independent chains for the same input."""

    chat_model = get_model()
    summary = (
        ChatPromptTemplate.from_template(
            "Summarize this topic in one sentence: {topic}"
        )
        | chat_model
        | parser
    )
    keywords = (
        ChatPromptTemplate.from_template(
            "Give three comma-separated keywords for: {topic}"
        )
        | chat_model
        | parser
    )

    chain = RunnableParallel(summary=summary, keywords=keywords)
    print("Parallel:", chain.invoke({"topic": "retrieval augmented generation"}))


def branch_chain() -> None:
    """Choose a chain based on an input value."""

    chat_model = get_model()
    question_chain = (
        ChatPromptTemplate.from_template("Answer this question briefly: {text}")
        | chat_model
        | parser
    )
    statement_chain = (
        ChatPromptTemplate.from_template("Rewrite this statement clearly: {text}")
        | chat_model
        | parser
    )

    chain = RunnableBranch(
        (lambda value: value["kind"] == "question", question_chain),
        statement_chain,
    )
    print(
        "Branch:",
        chain.invoke({"kind": "question", "text": "What is a chain?"}),
    )


def passthrough_and_lambda_chain() -> None:
    """Keep the original input and derive a second value from it."""

    normalize = RunnableLambda(lambda value: value.strip().lower())
    chain = RunnableParallel(
        original=RunnablePassthrough(),
        normalized=normalize,
    )
    print(
        "Passthrough and lambda:",
        chain.invoke("  LangChain Runnables  "),
    )


def main() -> None:
    sequence_chain()
    # parallel_chain()
    # branch_chain()
    # passthrough_and_lambda_chain()


if __name__ == "__main__":
    main()
