# LangChain chains

A chain connects multiple LangChain steps into one workflow. A step can be a
prompt, model, output parser, retriever, tool, or ordinary Python function.
The output of one step becomes the input to the next:

```text
input -> prompt -> model -> parser -> result
```

In modern LangChain, chains are usually built from **runnables** with LangChain
Expression Language (LCEL). Runnables share a common interface such as
`invoke`, `stream`, `batch`, and their async equivalents.

## Basic composition

The pipe operator (`|`) creates a sequence:

```text
step_a | step_b | step_c
```

This makes the data flow explicit and allows each step to be tested separately.
The final chain can be invoked as one runnable.

## Common chain patterns

| Pattern | What it does | Typical use |
| --- | --- | --- |
| **Sequence** | Runs steps in order; each output feeds the next step. | Prompt → model → parser. |
| **Parallel** | Runs independent steps for the same input and returns their results together. | Generate a summary and keywords at the same time. |
| **Branch** | Chooses a path based on the input or an earlier result. | Use different prompts for questions and commands. |
| **Runnable composition** | Preserves input or derives values with runnable helpers. | Use `RunnablePassthrough` and `RunnableLambda` inside a workflow. |

Parallel, branch, and helper workflows are represented as runnables, commonly
with `RunnableParallel`, `RunnableBranch`, `RunnablePassthrough`, and
`RunnableLambda`. 

## Chain inputs and outputs

Define the input and output shape of each step clearly. A chain may pass:

- a string, such as a user question;
- a dictionary, such as prompt variables or parallel results;
- an `AIMessage` from a chat model; or
- a typed value returned by an output parser.

Output parsers are useful at the end of a chain because they convert model
responses into strings, JSON, lists, or validated application objects.

## Runnable execution

Chains support the standard runnable methods:

- `invoke`: run one input and return the complete result;
- `stream`: receive output incrementally when supported;
- `batch`: process multiple inputs; and
- `ainvoke`, `astream`, and `abatch`: async equivalents.

Use `invoke` for normal execution. Use `stream` for user-facing progressive
responses and `batch` when processing independent inputs.

## Classic chains

Older LangChain versions exposed classes such as `LLMChain`,
`SimpleSequentialChain`, `SequentialChain`, `RouterChain`, and
`RetrievalQA`. These are useful when maintaining older code, but new workflows
are generally clearer when composed from prompts, models, retrievers, parsers,
and runnables directly.

## When to use a chain

Use a chain when the workflow has predictable steps and data flow. Agent
workflows, where the model decides which tools or steps to use dynamically,
are covered separately. Keep validation, permissions, retries, and external
side effects in application code around the chain rather than relying on the
model alone.

Examples for these patterns can be found in `playground.py`.
