from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI


EXAMPLES = [
    {
        "text": "The delivery arrived two days early.",
        "label": "positive",
    },
    {
        "text": "The package arrived damaged and unusable.",
        "label": "negative",
    },
    {
        "text": "The order arrived yesterday.",
        "label": "neutral",
    },
]


def create_example_prompt():
    """Define the format of one demonstration example."""
    return ChatPromptTemplate.from_messages(
        [
            ("human", "Classify the sentiment of this review: {text}"),
            ("ai", "{label}"),
        ]
    )


def create_few_shot_examples():
    """Create the reusable few-shot demonstration block."""
    return FewShotChatMessagePromptTemplate(
        examples=EXAMPLES,
        example_prompt=create_example_prompt(),
    )


def create_template():
    """Create a chat prompt containing demonstrations and a new input."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify review sentiment as exactly one of: "
                "positive, negative, or neutral.",
            ),
            create_few_shot_examples(),
            ("human", "Classify the sentiment of this review: {text}"),
        ]
    )


def build_prompt(text: str):
    """Render the demonstrations and a new review into chat messages."""
    return create_template().invoke({"text": text})


def run():
    prompt = build_prompt("The support team solved my issue immediately.")
    print("****************************")
    print(prompt)
    print("****************************")

    print("Formatted messages:")
    for message in prompt.to_messages():
        print(f"{message.type}: {message.content}")

    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(prompt)
    print("LLM response:", response.content)


if __name__ == "__main__":
    run()
