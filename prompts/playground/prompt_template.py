import json
from langchain_core.prompts import PromptTemplate
from langchain_core.load import dumps, load
from langchain_openai import OpenAI


def create_template_explicit():
    return PromptTemplate(
        template="Answer the following question: {question}",
        input_variables=["question"],
    )


#  Recommended way to create a prompt template from a template
def create_template_from_template():
    return PromptTemplate.from_template(
        "You are a {style} assistant. Answer: {question}"
    )


def build_prompt(question: str):
    """Build and return a formatted prompt for a given question."""
    prompt_template = create_template_explicit()
    return prompt_template.format(question=question)


def save_template(
    template: PromptTemplate, file_path: str = "prompts/templates/prompt_template.json"
):
    """Save the given prompt template to a JSON file."""
    prompt_dic = dumps(template)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt_dic)


def load_template(file_path: str = "prompts/templates/prompt_template.json"):
    """Load and return a prompt template from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        saved_prompt_dict = json.load(f)
    return load(saved_prompt_dict, allowed_objects="core")


def run_llm():
    llm_model = OpenAI(model="gpt-3.5-turbo-instruct")
    prompt = build_prompt("What is LangChain?")
    print("Formatted prompt:", prompt)
    response = llm_model.invoke(prompt)
    print("LLM response:", response)


def run_llm_with_style():
    llm_model = OpenAI(model="gpt-3.5-turbo-instruct")
    template = create_template_from_template()
    configured_prompt = template.partial(style="concise")
    prompt = configured_prompt.format(question="What is LangChain?")
    print("Formatted prompt:", prompt)
    response = llm_model.invoke(prompt)
    print("LLM response:", response)


if __name__ == "__main__":
    # run_llm()
    run_llm_with_style()
    # save_template(create_template_from_template())
