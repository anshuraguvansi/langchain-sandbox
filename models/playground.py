from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_protocol import Any
import asyncio
from sklearn.metrics.pairwise import cosine_similarity


class MonitorHandler(BaseCallbackHandler):
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print(f"response: {response}")
        print(f"kwargs: {kwargs}")
        print("LLM call ended.")


# *********************** LLM Models ***********************


def run_llm_sync():
    llm_model = OpenAI(model="gpt-3.5-turbo-instruct")
    response = llm_model.invoke("Tell me a joke.")
    print(response)


# *********************** Chat Models ***********************


def run_chat_sync(model: BaseChatModel = ChatOpenAI(model="gpt-4o-mini")):
    # Initialize the chat model and get a response from it.
    response = model.invoke(
        "Tell me a joke.",
        config={
            "run_name": "joke_generation",  # Custom name for this run
            "tags": ["humor", "demo"],  # Tags for categorization
            "metadata": {"user_id": "123"},  # Custom metadata
            "callbacks": [MonitorHandler()],  # Function to call with the response
        },
    )
    print(response.text)


# Run the asynchronous chat function
async def run_chat_async():
    chat_model = ChatOpenAI(model="gpt-4o-mini")
    system = SystemMessage(content="You are a good joke teller.")
    human = HumanMessage(content="Tell me a joke for a child.")
    response = await chat_model.ainvoke(
        [system, human],
        config={
            "run_name": "joke_generation",  # Custom name for this run
            "tags": ["humor", "demo"],  # Tags for categorization
            "metadata": {"user_id": "123"},  # Custom metadata
            "callbacks": [MonitorHandler()],  # Function to call with the response
        },
    )
    print(response.content)


async def run_chat_stream():
    chat_model = ChatOpenAI(model="gpt-4o-mini")
    system = SystemMessage(content="You are a good joke teller.")
    human = HumanMessage(content="Tell me a joke for a child.")
    async for response in chat_model.astream([system, human]):
        print(response.content)


async def run_chat_batch():
    chat_model = ChatOpenAI(model="gpt-4o-mini")
    system = SystemMessage(
        content="You are a helpful travel assistant. Keep answers concise."
    )
    questions = [
        "What are two must-see places in Kyoto?",
        "What is the best time of year to visit Iceland?",
        "Give me one useful tip for traveling around Portugal.",
    ]

    # Each item is one independent chat request. The requests can run together.
    requests = [[system, HumanMessage(content=question)] for question in questions]
    responses = await chat_model.abatch(requests)

    for question, response in zip(questions, responses):
        print(f"Question: {question}")
        print(f"Answer: {response.content}\n")


# *********************** Embeddings Models ***********************


def run_embedding_sync():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    documents = [
        "LangChain helps developers build applications powered by language models.",
        "Embedding models convert text into vectors for semantic search.",
        "Python lists preserve insertion order and can contain duplicate values.",
        "A chat model can answer questions using a sequence of messages.",
    ]
    query = "What is langchain?"

    document_vectors = embedding_model.embed_documents(documents)
    query_vector = embedding_model.embed_query(query)

    ranked_documents = sorted(
        zip(documents, document_vectors),
        key=lambda item: cosine_similarity([query_vector], [item[1]])[0][0],
        reverse=True,
    )

    print(f"Query: {query}\n")
    for document, document_vector in ranked_documents:
        score = cosine_similarity([query_vector], [document_vector])[0][0]
        print(f"{score:.4f} - {document}")


async def run_embedding_async():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=50)
    response = await embedding_model.aembed_query("What does LangChain do?")
    print(response)


## LLM models invocation
# run_llm_sync()

## Chat models invocation
# model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
# run_chat_sync(model=model)
# asyncio.run(run_chat_async())
# asyncio.run(run_chat_stream())
# asyncio.run(run_chat_batch())

# Embeddings models invocation
# run_embedding_sync()
# run_embedding_similarity()
# asyncio.run(run_embedding_async())
