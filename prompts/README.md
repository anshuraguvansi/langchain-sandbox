# LangChain Prompts

Prompts are the interface between application logic and a language model. They
convert structured application data—such as user requests, retrieved
documents, conversation history, tool results, and formatting requirements—into
the input consumed by a model.

This module focuses on the theory and design of prompts. Runnable Python
examples available in [playground](playground).

## Learning goals

After completing this module, you should understand:

- The difference between prompt templates, formatted prompts, and model input;
- The roles of `PromptTemplate`, `ChatPromptTemplate`, and related classes;
- How variables, partial variables, and optional message placeholders work;
- Why chat prompts represent roles and history explicitly;
- How prompts participate in LangChain Expression Language workflows;
- How prompts support retrieval, few-shot learning, structured output, and tool
  use;
- How to reason about context limits, trust boundaries, and prompt injection;
  and
- How to evaluate prompt quality independently from model quality.

## What is a prompt?

A prompt is the complete input for one model invocation. Depending on the model
interface, that input may be a single text string or an ordered collection of
messages.

It is useful to distinguish three concepts:

1. **Prompt template**: a reusable specification containing fixed instructions
   and named variable placeholders.
2. **Formatted prompt**: the concrete prompt produced after application values
   have been inserted into the template.
3. **Model response**: the output generated after the model receives the
   formatted prompt.

The template is application logic. The formatted prompt is request-specific
data. Keeping those concepts separate makes prompts easier to test, version,
inspect, and reuse.

## The prompt lifecycle

A typical prompt lifecycle has these stages:

1. **Definition**: establish instructions, message roles, variables, and output
   requirements.
2. **Input preparation**: collect and validate user input, retrieved context,
   history, examples, and tool results.
3. **Formatting**: render the template into a string or ordered messages.
4. **Inspection and policy checks**: verify required values, size limits,
   source boundaries, and sensitive-data rules.
5. **Model invocation**: pass the formatted prompt to the model.
6. **Output handling**: parse, validate, display, store, or route the response.
7. **Evaluation**: measure quality, cost, latency, safety, and regressions.

Prompt formatting is only one stage in this process. A well-written prompt
cannot correct missing data, irrelevant retrieval, an unsuitable model, or
missing authorization checks.

## Prompt templates in LangChain

Prompt templates are runnables. They accept named input values and return a
prompt value, which allows them to participate in composed LangChain workflows.

| Class | Input model | Output model | Primary purpose |
| --- | --- | --- | --- |
| `PromptTemplate` | Named values | Text prompt value | Completion-style or text-oriented models |
| `ChatPromptTemplate` | Named values | Ordered message prompt value | Chat models and role-aware workflows |
| `MessagesPlaceholder` | A message sequence | Inserted messages | Conversation history or dynamic messages |
| `FewShotPromptTemplate` | Example data and task input | Text containing demonstrations | Few-shot completion prompts |
| `FewShotChatMessagePromptTemplate` | Example data and task input | Demonstration messages | Few-shot chat prompts |

The runnable interface gives prompt templates common execution behavior with
other LangChain components. They can be invoked individually, streamed where
supported by a larger workflow, executed in batches, or composed with models,
retrievers, parsers, and application functions.

## Text prompts and chat prompts

`PromptTemplate` represents one text sequence. It is appropriate when the model
interface is built around a single string or when role information is not
needed.

`ChatPromptTemplate` represents an ordered conversation. The order and role of
each message are part of the input's meaning, not merely presentation details.
Chat prompts are generally preferable for modern chat models because they make
system instructions, user requests, assistant demonstrations, tool results,
and history explicit.

Common message roles include:

- **System**: application-level behavior, constraints, policies, and task
  framing.
- **Human**: the current user request or an application-provided request.
- **AI**: an earlier assistant response, usually used for history or examples.
- **Tool**: a result produced by an external tool and returned to the model.

Provider support and behavior can vary. The application should confirm which
roles, content types, and multimodal inputs its selected model supports.

## Variables and input contracts

Variables define the contract between application code and a prompt. A prompt
should make clear which values are required, which values are optional, and
which values are fixed by the application.

The default template syntax uses named placeholders. Missing required values
should fail during formatting rather than silently producing an incomplete
request. Explicit validation is especially important when prompts are reused by
multiple application paths.

Literal formatting characters must be distinguished from variable markers.
This matters for structured formats such as JSON, XML, or other content that
uses braces. User-provided values should be inserted as data and should not be
interpreted as a second template.

### Partial variables

Partial variables are values bound when a prompt is configured rather than when
each request is processed. They are useful for stable instructions, shared
formatting rules, application metadata, or context generated by a known
function.

Partials should not hide important request inputs. Values that vary per request,
especially user-controlled values, should remain visible in the prompt's input
contract. This separation improves testing, tracing, and security review.

### Optional messages

`MessagesPlaceholder` provides a controlled location for a dynamically supplied
message sequence. It is useful for conversation history, retrieved
conversation context, or tool messages whose count is not known when the
template is defined.

An optional placeholder allows a caller to omit that message sequence. A
required placeholder expresses that the workflow is incomplete without it.
This distinction prevents accidental assumptions about memory and conversation
state.

## Conversation history is application state

Chat models do not automatically remember previous invocations. History must be
loaded and supplied by the application or by a state-management layer.

History management includes deciding:

- which turns remain relevant;
- whether old turns should be removed, summarized, or retrieved selectively;
- how much context budget should be reserved for the current request and answer;
- how sensitive information should be redacted; and
- which messages belong to which trust level.

Unbounded history increases cost, latency, and the chance that important
instructions or facts are diluted by irrelevant content. Memory is therefore a
data-management problem as well as a prompt-design problem.

## Prompt composition and LCEL

LangChain Expression Language treats prompts as composable workflow steps. A
common conceptual pipeline is:

```text
input values -> prompt -> model -> output parser -> application result
```

The prompt establishes the model-facing contract. The model produces a message
or text response, and the parser converts that response into the type expected
by the application.

Prompt composition enables reusable workflows such as:

- translation, classification, summarization, or extraction;
- retrieval-augmented generation, where retrieved documents become bounded
  context;
- tool workflows, where tool results are inserted as messages; and
- multi-step pipelines, where one prompt's output becomes another step's input.

Composition should preserve clear boundaries between steps. Each prompt should
have a focused responsibility and a documented input and output contract.

## Few-shot prompting

Few-shot prompting provides demonstrations of the task to influence the
model's behavior. Examples can communicate formatting, label meanings, tone,
reasoning patterns, and edge-case handling more precisely than instructions
alone.

Effective demonstrations are:

- representative of real inputs;
- consistent with the desired output format;
- diverse enough to cover important cases;
- small enough to leave room for the request and response; and
- reviewed for accidental bias, incorrect answers, and sensitive data.

More examples do not necessarily improve quality. Examples consume context and
can cause the model to imitate irrelevant details. Dynamic example selection
should be evaluated just like retrieval.

## Retrieval and context design

In retrieval-augmented generation, the prompt connects retrieval to generation.
It should define where context appears, how it relates to the question, what
sources are authoritative, and what the model should do when the context does
not contain an answer.

Retrieved content is evidence, not automatically trusted instruction. Prompts
should distinguish application instructions from external context and should
encourage grounded answers without implying that missing information may be
invented.

Prompt quality cannot compensate for poor retrieval. Evaluate retrieval
relevance, context completeness, answer groundedness, and generation quality as
separate but related concerns.

## Structured output

Natural-language responses are often unsuitable for downstream application
logic. Structured output uses a defined schema, model capability, and/or output
parser to produce predictable fields.

A prompt can describe the desired fields, types, allowed values, and fallback
behavior. However, a textual instruction to produce JSON is not validation.
Applications should validate parsed output, handle malformed responses, enforce
field constraints, and decide whether retries or human review are appropriate.

Structured output is particularly useful for classification, extraction,
routing, tool arguments, and workflow state transitions.

## Prompt security and injection

Prompt injection occurs when content supplied to the model attempts to change
the application's intended behavior. It can originate in user input, uploaded
documents, web pages, retrieved records, tool results, or conversation history.

Important principles include:

- treat external text as data rather than trusted application instruction;
- keep system-level instructions conceptually separate from user and retrieved
  content;
- define source authority and conflict behavior;
- never place secrets, API keys, or credentials in prompts;
- never let model text alone authorize sensitive actions;
- validate tool names, arguments, permissions, and side effects in code;
- use approval or human review for high-impact operations; and
- minimize sensitive prompt logging while retaining enough metadata to debug.

A system message is a communication mechanism, not a security boundary.
Authorization and data protection must be enforced outside the model.

## Context, tokens, and cost

The context sent to a model may include system instructions, examples, history,
retrieved documents, tool results, the current request, and output-format
requirements. All of these compete for the model's context window.

Large prompts can increase cost and latency, exceed model limits, reduce the
relative importance of key instructions, and lower answer quality. Prompt
design therefore includes context budgeting:

- reserve space for the model's response;
- retrieve only relevant content;
- remove duplicated instructions and repeated context;
- summarize or selectively retain history;
- keep examples compact; and
- measure tokens rather than estimating size from character count alone.

Token limits, pricing, message serialization, and supported features vary by
provider and model. Treat them as configuration that should be measured and
verified for the deployed model.

## Prompt evaluation and lifecycle management

Prompt quality should be evaluated on a representative dataset rather than a
single successful interaction. Include normal, ambiguous, incomplete, long,
malformed, and adversarial inputs.

Evaluate:

- correctness and completeness;
- adherence to output format or schema;
- groundedness in supplied context;
- appropriate uncertainty and fallback behavior;
- safety and resistance to instruction manipulation;
- latency, token usage, and cost; and
- regressions against previously passing cases.

Good prompt lifecycle practices include:

1. Keep prompt definitions in source control.
2. Give prompts stable names and document their input contracts.
3. Test formatting without making a model call.
4. Inspect rendered prompts and messages during development.
5. Test chains with fake or deterministic models where possible.
6. Version prompts with relevant model and parser configuration.
7. Record why a prompt changed and which evaluation cases motivated the change.

Observability should expose prompt versions, inputs' non-sensitive metadata,
latency, token usage, and errors. Avoid storing raw sensitive content by
default.

## Common mistakes

- Choosing a text prompt when the model workflow requires role-aware messages.
- Treating a chat model's response message as though it were already a plain
  string or validated object.
- Omitting required variables or silently treating important data as optional.
- Forgetting that literal formatting characters may need escaping.
- Formatting user-provided text twice as though it were template source.
- Assuming conversation history or memory exists automatically.
- Sending all available documents instead of relevant, bounded context.
- Putting unrelated tasks and instructions into one oversized prompt.
- Relying on prompt wording for authorization, schema validation, or safety.
- Copying outdated chain patterns without checking the installed LangChain
  version and provider capabilities.

## Playground and exercises

Implementation examples and experiments belong in [playground.py](playground.py).
Use that file to explore the concepts in this README, including:

1. Defining and inspecting text and chat prompt templates.
2. Comparing required variables, partial variables, and optional history.
3. Rendering few-shot prompts and measuring their context cost.
4. Composing a prompt with a model and an output parser.
5. Building a retrieval-oriented prompt with explicit context boundaries.
6. Testing malformed inputs, missing context, and prompt injection attempts.
7. Comparing prompt versions with a small evaluation dataset.


