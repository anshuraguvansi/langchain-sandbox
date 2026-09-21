"""First LangChain playground: compose a prompt, model, and output parser."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main() -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You explain technical topics clearly for a beginner. "
                "Keep the answer concise.",
            ),
            ("human", "Explain {topic} in three short sentences."),
        ]
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | model | StrOutputParser()

    result = chain.invoke({"topic": "LangChain"})
    print(result)


if __name__ == "__main__":
    main()
