import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.load import dumps, load
from langchain_openai import ChatOpenAI


def create_template_explicit():
    return ChatPromptTemplate(
        messages=[
            ("system", "You are a helpful {style} assistant."),
            ("human", "{question}"),
        ],
        input_variables=["style", "question"],
    )


# Recommended way to create a chat prompt template from messages
def create_template_from_template():
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {style} assistant."),
            ("human", "{question}"),
        ]
    )


def build_prompt(style: str, question: str):
    """Build and return a formatted prompt for a given question."""
    prompt_template = create_template_from_template()
    return prompt_template.format(style=style, question=question)


def save_template(
    template: ChatPromptTemplate,
    file_path: str = "prompts/templates/chat_prompt_template.json",
):
    """Save the given prompt template to a JSON file."""
    prompt_dic = dumps(template)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt_dic)


def load_template(file_path: str = "prompts/templates/chat_prompt_template.json"):
    """Load and return a prompt template from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        saved_prompt_dict = json.load(f)
    return load(saved_prompt_dict, allowed_objects="core")


# inserting a dynamic list of messages, usually conversation history.
def create_template_with_history():
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}"),
        ]
    )


def run():
    llm_model = ChatOpenAI(model="gpt-4o-mini")
    template = create_template_from_template()
    configured_prompt = template.partial(style="concise")
    prompt = configured_prompt.format(question="What is LangChain?")
    print("Formatted prompt:", prompt)
    response = llm_model.invoke(prompt)
    print("LLM response:", response)


def run_with_history():
    llm_model = ChatOpenAI(model="gpt-4o-mini")
    template = create_template_with_history()
    configured_prompt = template.partial(style="concise")
    prompt = configured_prompt.format(
        question="What is LangChain?",
        history=[
            HumanMessage(content="What is LangChain?"),
            AIMessage(
                content="LangChain is a framework for building LLM applications."
            ),
        ],
    )
    print("Formatted prompt:", prompt)
    response = llm_model.invoke(prompt)
    print("LLM response:", response)


if __name__ == "__main__":
    # run()
    run_with_history()
    # save_template(create_template_from_template())
