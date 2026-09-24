# VyaparSathi AI Agent Documentation

## 1. Overview

This project contains an AI agent built for a retail operations assistant called Vyapar Copilot. The agent is designed to answer questions related to:

- inventory health
- stockouts and low-stock alerts
- sales summaries
- top-selling products
- forecasted demand
- restock prioritization
- operational insights
- store-specific memory and user preference memory

The agent is implemented as a LangGraph workflow, with a FastAPI API layer in front of it. It uses MongoDB for checkpointing and persistent conversation state, and Mem0 for long-term memory.

This document is written so another AI system can understand the project deeply enough to suggest improvements, refactors, and architecture changes correctly.

---

## 2. High-level architecture

### Core components

1. FastAPI service
   - Exposes endpoints for agent requests
   - Handles HTTP auth context (`x-user-id`)
   - Creates and manages the app lifecycle
   - Makes the LangGraph agent accessible to the frontend

2. LangGraph agent graph
   - Uses `StateGraph` to define the agent workflow
   - Controls looped reasoning and tool calls
   - Maintains state across the full conversation

3. State container
   - `VyaparAgentState` is the central runtime state object
   - Stores user prompt, goal, tool results, memory flags, messages, and final answer

4. Tool layer
   - Domain-specific tools for inventory, sales, forecast, insights, and clarification
   - Each tool queries MongoDB-backed store data

5. Memory system
   - `mem0` stores user preferences and store knowledge
   - Used for long-term personalization and contextual recall

6. Checkpointer
   - MongoDB-backed LangGraph saver stores checkpoints by `thread_id`
   - Enables interrupted workflows and chat persistence

7. LLM provider abstraction
   - Built around a fallback provider chain
   - Uses Gemini, Groq, NVIDIA providers with automatic failover

---

## 3. Major project structure relevant to the agent

### Main agent package

- `app/agent/__init__.py`
  - Exposes public agent APIs and modules

- `app/agent/graph.py`
  - Defines the graph and node routing logic

- `app/agent/state.py`
  - Defines `VyaparAgentState`, tool result structures, and initial state factory

- `app/agent/checkpointer.py`
  - Creates and manages the MongoDB LangGraph checkpointer

- `app/agent/memory.py`
  - Mem0 memory management and retrieval logic

- `app/agent/utils.py`
  - Shared utility functions for message extraction and formatting

### Node directory

- `app/agent/nodes/grader_node.py`
  - Safety / scope guardrail for user prompt

- `app/agent/nodes/intent_node.py`
  - Classifies intent: live_data, memory, recap, mixed, general
  - Determines if the request is complex and needs planning

- `app/agent/nodes/planner_node.py`
  - Generates a structured execution plan for complex requests without executing tools

- `app/agent/nodes/think_node.py`
  - Main reasoning node
  - Decides tools or completion

- `app/agent/nodes/memory_query.py`
  - Fetches persistent memory from mem0 when needed

- `app/agent/nodes/tool_node.py`
  - Invokes selected tools

- `app/agent/nodes/observe_node.py`
  - Summarizes tool outputs and appends them as `ToolMessage`

- `app/agent/nodes/interrupt_node.py`
  - Pauses the graph with `interrupt()` when more info is needed

### Tools directory

- `app/agent/tools/clarification/__init__.py`
  - `ask_for_clarification`

- `app/agent/tools/inventory/__init__.py`
  - `get_inventory_summary`
  - `get_low_stock_products`

- `app/agent/tools/sales/__init__.py`
  - `get_sales_summary`
  - `get_top_selling_products`

- `app/agent/tools/forecast/__init__.py`
  - `get_restock_priorities`
  - `get_forecast_summary`

- `app/agent/tools/insights/__init__.py`
  - `get_store_insights`

- `app/agent/tools/registry.py`
  - Registers all tools used by the graph

### Service layer

- `app/agent/service/inventory/__init__.py`
  - Inventory aggregation queries against MongoDB

- `app/agent/service/sales/__init__.py`
  - Sales aggregation and summary queries

- `app/agent/service/forecast/__init__.py`
  - Restock priority and forecast logic

---

## 4. Graph structure and execution flow

The graph is built in `app/agent/graph.py` using `StateGraph(VyaparAgentState)`.

### Entry flow

1. `grader` node runs first.
2. If the prompt is denied as harmful, the flow ends with a refusal answer.
3. Otherwise it moves to `intent`.

### Main loop

The standard flow is:

`grader -> intent -> [planner if complex] -> think -> (interrupt OR memory_query OR tool) -> observe -> think ...`

### Routing logic

In `graph.py`:

- after `grader`, route is either:
  - `END` if denied
  - `intent` if safe

- after `intent`, route is:
  - `planner` if the request is complex (`requires_planning=True`)
  - `think` if the request is simple

- after `think`, route priority is:
  1. `interrupt` if `needs_clarification` is true
  2. `memory_query` if `memory_query_needed` is true
  3. `tool` if tool calls are pending
  4. `END` if `goal_status == "complete"`
  5. `END` otherwise

This gives the agent a looped reasoning system where it can:

- think
- decide on a tool
- fetch real data
- reason again
- ask a clarification if needed
- finish with an answer

---

## 5. State model (`VyaparAgentState`)

`VyaparAgentState` is a TypedDict that stores everything required to run one agent interaction.

### Important state sections

#### Identity and scope

- `user_id`
- `store_id`
- `thread_id`
- `user_prompt`

These define which user, which store, and which logical chat session the agent is operating in.

#### Goal tracking

- `goal`
- `goal_status`
- `loop_count`
- `max_loops`
- `requires_planning`
- `plan`

The agent loop is bounded to prevent infinite loops. Default `max_loops` is 3.
The `plan` stores a structured execution strategy for complex goals.

#### Message history

- `messages`

This uses `add_messages` from LangGraph, so messages accumulate safely across the graph lifecycle.

#### Tool execution tracking

- `pending_tool_calls`
- `tool_results`
- `tools_called_this_loop`

The `think` node decides tools. The `tool` node executes them. The `observe` node converts results into `ToolMessage` entries for the model.

#### Context buckets

- `inventory_context`
- `sales_context`
- `forecast_context`
- `restock_context`
- `insights_context`

These are not just final text—they are the structured data storage containers for gathered business facts.

#### Long-term memory

- `user_preferences`
- `store_knowledge`
- `user_memory_loaded`
- `store_memory_loaded`
- `memory_query_needed`

This is where mem0 memory is merged into the conversation.

#### Final output

- `final_answer`
- `response_metadata`
- `error`
- `should_persist_memory`

---

## 6. The graph nodes explained in detail

### 6.1 `grader_node`

The starter guardrail checks if the user prompt is unsafe or off-topic.

Responsibilities:

- read latest human prompt
- skip classification if resuming after clarification
- run safety keyword screening
- allow non-harmful prompts through
- set `grader_denied` when necessary

Important behavior:

- it is intentionally permissive now except for clearly harmful keywords
- it does not block all business questions in the current state

This is a useful place to improve later with stricter intent classification or prompt moderation.

### 6.2 `intent_node`

This node decides the nature of the request using deterministic keyword matching and summarization-based intent extraction.

Possible intents:

- `live_data`
- `memory`
- `conversation_recap`
- `mixed`
- `general`

The node sets `memory_query_needed` for memory-based or mixed requests.
It also sets `requires_planning` to `True` for complex intents (e.g. "recommend strategy", "which products should I restock").

This is important because the agent does not always need memory or a plan. It only uses them when relevant.

### 6.3 `planner_node`

This node is used for complex requests to generate a step-by-step strategy.

Responsibilities:

- read the user prompt
- build a 2 to 6 step plan mapping to available tools
- format the output as JSON containing `goal` and `steps`
- save the output to `state["plan"]`

Important behavior:

- it does NOT execute tools itself
- it acts as a strategizer to guide the `think_node` through a multi-step process
- it only runs once per request when routed from `intent`

### 6.4 `think_node`

This is the main decision-making node.

It:

- looks at current goal and loop number
- determines if memory is required
- loads the LLM
- binds tools to the model if allowed
- builds system prompt instructions
- injects memory context
- injects tool synthesis context from previous loops
- preserves recent chat history with a sliding window
- decides:
  - call tool(s)
  - ask clarification
  - mark goal complete
  - stop due to loop limit

Important logic:

- It uses the `plan` from the `planner_node` (if present) to sequence its tool calls effectively.
- If `loop_count >= max_loops`, it forces a direct-to-answer path instead of more tools.
- If the model wants `ask_for_clarification`, it does not execute a tool; it triggers the interrupt route instead.
- It tries to preserve conversation continuity while avoiding context bloat.

### 6.5 `memory_query_node`

This node loads persisted memory from mem0 when the request needs context from previous conversations or user preferences.

It fetches:

- user memory (preferences, tone, language, style)
- store memory (facts, patterns, decisions, observations)

It sets:

- `user_memory_loaded = True`
- `store_memory_loaded = True`
- `memory_query_needed = False`

### 6.6 `tool_node`

This node executes any tool calls the LLM requested.

It:

- reads `pending_tool_calls`
- resolves tool by name from registry
- injects `store_id` if missing
- calls `tool.ainvoke(args)`
- stores results in `tool_results`
- routes structured data into relevant context buckets

Examples:

- inventory tools -> `inventory_context`
- sales tools -> `sales_context`
- forecast tools -> `forecast_context`
- insights tool -> `insights_context`

### 6.7 `observe_node`

The observe node transforms raw tool results into `ToolMessage` objects that the LLM can reason over.

Important behavior:

- only results produced in the current loop are converted
- large payloads are summarized before being passed to the LLM
- actual full data remains in state buckets and is not discarded

This is crucial for controlling token usage and preserving context quality.

### 6.8 `interrupt_node`

This node is used when the agent is uncertain and needs clarification before continuing.

It:

- generates a specific question
- calls `langgraph.types.interrupt()`
- pauses execution and saves graph state
- resumes when the user provides the answer
- stores the Q&A in `clarification_history`

This is a very important design feature for agent reliability when the request is ambiguous.

---

## 7. Tool system

### Tool registry

`app/agent/tools/registry.py` defines the available readable tools.

The registry includes:

- `ask_for_clarification`
- `get_inventory_summary`
- `get_low_stock_products`
- `get_sales_summary`
- `get_top_selling_products`
- `get_restock_priorities`
- `get_forecast_summary`
- `get_store_insights`

### Tool responsibilities

#### `get_inventory_summary`

Gets overall inventory statistics:

- total products
- low stock count
- out-of-stock count
- total inventory value
- low stock threshold used

#### `get_low_stock_products`

Returns individual products that are nearing stock-out thresholds.

#### `get_sales_summary`

Returns total revenue, total sales count, and average order value over a lookback period.

#### `get_top_selling_products`

Returns the best-performing products by sales quantity and revenue.

#### `get_restock_priorities`

Computes Red/Yellow/Green restock urgency using:

- current quantity
- average daily sales
- days to stockout
- lead time
- safety stock buffer

#### `get_forecast_summary`

Returns predicted demand for a time horizon and tracks likely stockout risk.

#### `get_store_insights`

Uses AI-driven insights logic to surface operational issues like:

- dead stock risk
- low-stock pressure
- out-of-stock products
- high-performing leaders

#### `ask_for_clarification`

Used when the user's request is too vague to answer confidently.

---

## 8. Service layer logic

The tools do not directly do heavy logic; they call service functions in `app/agent/service`.

### Inventory service

`app/agent/service/inventory/__init__.py`

It aggregates the MongoDB `products` collection and gives:

- overall product count
- low-stock threshold calculations
- money value of stock
- low-stock item list

### Sales service

`app/agent/service/sales/__init__.py`

It aggregates the MongoDB `sales` collection using `completedAt` and product item data to compute:

- revenue totals
- order counts
- average order value
- top products by quantity and revenue

### Forecast service

`app/agent/service/forecast/__init__.py`

It calculates:

- average daily sales
- days to stockout
- suggested restock amounts
- priority categories
- forecast trend values

This is a key operational layer for the agent.

---

## 9. LLM provider system

The project is designed to be resilient to provider outages and rate limits.

File: `app/lib/llm.py`

### Fallback order

The system attempts:

1. Gemini 2.5 Flash
2. Groq GPT-OSS 120B
3. NVIDIA Llama 3.1 8B
4. Groq GPT-OSS 20B
5. Gemini 2.5 Flash Lite

The reason for this design is that free-tier quotas can be exhausted quickly, especially for Gemini. The fallback chain prevents total failure when one provider is unavailable.

### Why it matters

The agent is not just one model. It is a provider abstraction with resilience handling, which is very important for production-grade AI systems.

---

## 10. Persistence and chat memory

### 10.1 LangGraph checkpointing

The `checkpointer.py` file creates a MongoDB-based LangGraph saver.

This is used to persist graph state and allow resumed execution after clarification.

Key idea:

- each chat has its own `thread_id`
- `thread_id` usually looks like `user_id:store_id:chat_id`
- the graph state can be continued across interaction boundaries

### 10.2 Chat sessions

The app integrates with chat session management in the route layer and services for conversation history.

This enables:

- multiple chat threads per user/store
- message persistence
- resumed conversations
- better user session continuity

### 10.3 Long-term memory (mem0)

`app/agent/memory.py` integrates with Mem0 for persistent memory.

It stores:

- user-level memory via `user_id`
- store-level memory via `store_id`

#### User memory examples

- preferred language
- tone preference
- reply detail level
- business preferences

#### Store memory examples

- frequent product patterns
- seasonal demand patterns
- previous decisions
- store-specific operational notes

Important design rule:

- memory is not treated as live data
- it is only context for reasoning
- real numbers still come from tools and MongoDB queries

---

## 11. API integration

The main route file is:

- `app/routes/store_ai_routes.py`

### Relevant endpoint

- `POST /{store_id}/copilot/stream`

This endpoint:

1. reads `x-user-id` from the request headers
2. resolves or creates a chat session
3. creates a fresh graph and initial state
4. streams SSE updates from the graph
5. handles interruptions and clarification
6. persists the final result and memory in the background

### Message flow

The API uses a streaming SSE model to communicate progress including:

- token generation
- tool execution events
- tool result updates
- memory status events
- clarification requests

This is ideal for frontend UX because the user can see the agent working in real time.

---

## 12. Initial state creation

The initial state factory is in `app/agent/state.py`.

It sets defaults such as:

- `goal_status = "pending"`
- `loop_count = 0`
- `max_loops = 3`
- `messages = [HumanMessage(content=user_prompt)]`
- empty context buckets
- memory flags false

The `thread_id` is important because it is how the service separates distinct chat windows.

---

## 13. Core reasoning model

The agent’s theoretical operating pattern is:

1. Understand the user prompt
2. Determine whether it is live-data, memory-based, recap-based, or general
3. Decide whether a tool is needed
4. Pull data from MongoDB or mem0
5. Combine the data with memory and prior context
6. Ask for clarification if necessary
7. Produce a final answer

This is not a static chatbot; it is a tool-using research assistant for store operations.

---

## 14. Why this agent is designed this way

This architecture was chosen because retail assistant tasks usually require:

- live business data
- historical context
- memory of user preferences
- ambiguity handling
- controlled reasoning loops

If the system did not use tools and memory, it would be unable to answer factual questions like:

- Which products are low stock?
- Which products should be reordered first?
- Which products are currently trending?
- What was my previous preference for report formatting?

The design intentionally separates:

- state
- memory
- tools
- LLM reasoning
- runtime execution

This makes the system easier to debug and improve.

---

## 15. Important design strengths

### 15.1 Good separation of concerns

The code neatly separates:

- state (`state.py`)
- orchestration (`graph.py`)
- LLM logic (`think_node.py`)
- data access (`service/`)
- orchestration memory (`memory.py`)

### 15.2 Tool-first architecture

The model is forced to operate through tools instead of guessing from hidden knowledge.

This yields more factual answers and less hallucination for operational tasks.

### 15.3 Memory-aware behavior

User preferences and store patterns are carried across chats and future requests.

### 15.4 Good loop control

The graph avoids infinite cycling via `loop_count` and `max_loops`.

### 15.5 Production readiness features

- MongoDB checkpointing
- SSE streaming API
- provider fallback
- structured tool outputs
- memory persistence

---

## 16. Current limitations and weak spots

This agent is good, but there are still important gaps that can be improved.

### 16.1 Safety classification is intentionally relaxed

In `grader_node.py`, the code says the scope classification is temporarily disabled.

This means:

- broad prompts may not be filtered as strictly as desired
- future harmful/off-topic prompts could slip through if the guardrail is not tightened

### 16.2 Tool coverage could be expanded

The system currently mostly handles retail operations data, but more advanced tasks could be added:

- product-level demand modeling
- supplier performance insights
- employee activity summaries
- profitability analysis
- shelf-life monitoring
- markdown or CSV export generation

### 16.3 Memory writing is opportunistic

Memory persistence is controlled by a `should_persist_memory` flag and a summarization strategy. This is good, but it can produce noisy or low-quality memory if not carefully managed.

### 16.4 Prompt tuning is critical

The quality of the final answer depends heavily on the system prompt and tool synthesis context. Small prompt changes can noticeably alter behavior.

### 16.5 Tool outputs may become large

Large tool results are summarized for the LLM, but this is a tradeoff between context efficiency and information completeness.

### 16.6 Error handling could be more structured

Some tool and provider failures are logged but the agent does not always recover elegantly or present a “why this failed” explanation cleanly.

---

## 17. Improvement opportunities

Here are realistic upgrade ideas for another AI to evaluate.

### A. Better guardrails

Add:

- stronger intent validation
- prompt classification before the agent runs
- explicit refusal categories and policy handling

### B. Stronger tool orchestration

Improve:

- tool selection reliability
- multi-tool sequencing logic
- better explanations of why a tool was chosen

### C. Better summarization quality

The system uses summarization on large outputs. Improvement areas:

- product-specific summarization
- better business KPI compression
- better detection of signal versus noise

### D. More advanced memory

Possible improvements:

- semantic memory retrieval
- store-specific memory summaries
- memory conflict detection
- user preference expiry rules

### E. Conversational quality

Improve:

- tone control
- answer formatting for various businesses
- trend explanation quality
- recommendation ranking

### F. Observability

Add:

- trace IDs for each run
- tool execution timing logs
- prompt template version metadata
- runtime diagnostic dashboard

---

## 18. Example questions to ask another AI for improvements

Below are copy-paste prompts you can use with another AI to improve this agent.

### Prompt 1: architecture review

> Review this VyaparSathi AI agent architecture. It is a LangGraph workflow with FastAPI, MongoDB checkpointer, Mem0 long-term memory, and domain tools for inventory, sales, forecast, and insights. Identify the biggest improvement opportunities in architecture, resilience, observability, and tool orchestration. Return a prioritized list with reasoning and concrete code-level recommendations.

### Prompt 2: performance optimization

> Analyze this LangGraph retail assistant for token efficiency, tool-call efficiency, and context-window optimization. Explain where the current design may waste tokens or increase latency, and suggest specific improvements with examples tied to this codebase.

### Prompt 3: reasoning quality improvement

> Look at the `think_node` and tool orchestration in this agent. Suggest ways to improve the agent’s ability to decide which tools to call, when to ask clarifying questions, and how to structure the final answer for business users.

### Prompt 4: memory improvement

> Please evaluate the Mem0 integration in this project. Suggest a better memory architecture for user preferences and store-specific knowledge, including retrieval quality, deduplication, and memory staleness handling.

### Prompt 5: production hardening

> Review this AI agent for production readiness. Focus on failures in LLM provider fallback, guardrail logic, MongoDB checkpointing, error recovery, and long-running conversation state. Propose a hardened architecture with clear responsibilities and safer defaults.

### Prompt 6: agent refactor

> Re-architect this agent into a cleaner, more maintainable system while preserving behavior. Suggest a revised folder structure, clearer node responsibilities, improved state modeling, and better separation between memory, tools, and orchestration.

### Prompt 7: debugging assistance

> I have a LangGraph-based retail copilot. Explain how to debug interrupt handling, tool-call mismatches, LLM tool call issues, and state contamination across conversation turns. Use this project’s actual file names and architecture as context.

---

## 19. Suggested “full context” prompt for future AI conversations

Use this when asking another AI for a detailed review:

> I’m working on a Python retail AI agent built with FastAPI and LangGraph. The project structure includes `app/agent/graph.py`, `state.py`, `nodes/*`, `tools/*`, `service/*`, `memory.py`, and a MongoDB-backed LangGraph checkpointer. The agent routes user requests through a grader, intent classifier, think node, memory fetch, tool execution, observe step, and optional clarification interrupt. Tools query store inventory, sales, forecast, and insight data from MongoDB. Long-term memory is handled via Mem0 for user preferences and store knowledge. Please analyze the architecture, explain its strengths and weaknesses, propose improvements, and suggest code-level refactors while preserving the current business intent.

---

## 20. Best way to use this document

This file should be treated as:

- architecture context for AI review
- project memory for future improvements
- a basis for prompt engineering work
- a document for debugging and refactoring planning

When you ask another AI for improvements, include:

1. the purpose of the agent
2. the actual file names involved
3. the main LLM and memory technology used
4. the tool categories and data sources
5. the specific problem you want fixed
6. the expected output of the new design

---

## 21. Short summary

This agent is a domain-specific retail operations assistant built with:

- LangGraph for orchestration
- FastAPI for API access
- MongoDB for state and data persistence
- Mem0 for memory
- domain tools for real store operations
- provider fallback LLM model selection

It is a well-structured tool-driven agent intended to answer operational questions using real data rather than general AI guesses.

The strongest next improvements are in:

- guardrail quality
- better tool orchestration
- memory precision
- observability
- production hardening

---

## 22. Quick map of the most important files

- `app/agent/graph.py` — graph orchestration
- `app/agent/state.py` — central state model
- `app/agent/nodes/think_node.py` — reasoning and tool selection
- `app/agent/nodes/tool_node.py` — execution of tools
- `app/agent/nodes/observe_node.py` — summarized event output
- `app/agent/nodes/interrupt_node.py` — clarification handling
- `app/agent/memory.py` — long-term memory integration
- `app/agent/checkpointer.py` — checkpoint persistence
- `app/lib/llm.py` — LLM provider fallback
- `app/routes/store_ai_routes.py` — API layer
- `app/agent/tools/*` — domain tools (as domain packages)
- `app/agent/service/*` — DB data access logic (as domain packages)

This document is intended to be a precise reference so that future AI systems can reason about the project correctly and suggest meaningful improvements.
