from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAI


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
    """Define the text format of one demonstration example."""
    return PromptTemplate.from_template("Review: {text}\nSentiment: {label}")


def create_template():
    """Create a text prompt containing examples and a new review."""
    return FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=create_example_prompt(),
        prefix=(
            "Classify each review as exactly one of: "
            "positive, negative, or neutral.\n\n"
        ),
        suffix="Review: {text}\nSentiment:",
        input_variables=["text"],
    )


def build_prompt(text: str):
    """Render the demonstrations and a new review into one text prompt."""
    return create_template().format(text=text)


def run():
    prompt = build_prompt("The support team solved my issue immediately.")

    print("Formatted prompt:")
    print(prompt)

    model = OpenAI(model="gpt-3.5-turbo-instruct")
    response = model.invoke(prompt)
    print("LLM response:", response)


if __name__ == "__main__":
    run()
