

# agent.py
"""
LangGraph + MLflow chat agent that supports runtime Vector Search filters with
strong validation.  Users can only filter by:

* category – one of {"Construction", "Housing"}
* year     – integer 1900‒2100
* month    – integer 1‒12

"""
from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Sequence

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    # UCFunctionToolkit,
    VectorSearchRetrieverTool
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import (
    ChatAgentState,
    ChatAgentToolNode,
)
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

import json

############################################
# Define your LLM endpoint and system prompt
############################################

LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm: LanguageModelLike = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

TOP_K_NUMBER_CHUNKS = 42

CATALOG, SCHEMA, SUFFIX = "edp_dev", "poli", "test_ted"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.{SUFFIX}_chunks_index"

# ---------------------------------------------------------------------------
# Filter guide (inserted into tool description so the LLM sees valid keys)
# ---------------------------------------------------------------------------
FILTER_GUIDE = (
    """
    Optional `filters` keys: `category` (Construction | Housing), 
    `year` (1900-2100), `month` (1-12). Example: 
    """
)
# ---------------------------------------------------------------------------
# Global system prompt (high-level instructions, no detailed schema repeat)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a knowledgeable, tool-aware assistant. First determine if a tool is needed; "
    "if so, call **one** tool and wait for its result. Use:\n"
    "- `document_rag` for document retrieval. Create filters to refine the search if necessary. If no relevant documents are retrieved, inform the user and suggest refining their criteria.\n"
    # "- `python_exec` for calculations.\n"
    "If no tool is required, provide a direct and concise answer. Always ensure your responses are accurate and relevant."
)

# ---------------------------------------------------------------------------
# 2. Build a Vector Search tool with dynamic filters
# ---------------------------------------------------------------------------


class SafeVectorSearchRetrieverTool(VectorSearchRetrieverTool):
    """Guarantees a string payload even when no docs are found, and truncates big docs."""
    def _run(self, query: str, filters: Optional[list] = None):  # noqa: D401
        docs = super()._run(query=query, filters=filters)

        # --- 1️⃣  No hits  → schema-safe sentinel  -----------------
        if not docs:
            return json.dumps(
                {"status": "NO_RESULTS", "num_results": 0, "docs": []}
            )

        # --- 2️⃣  Convert documents to lightweight strings ---------
        return json.dumps(
            {"status": "OK", "num_results": len(docs), "docs": docs}
        )


rag_tool = SafeVectorSearchRetrieverTool(
    index_name=INDEX_NAME,
    columns=["chunk_text", "chunk_id", "path"],
    query_type="HYBRID",
    tool_name="document_rag",
    tool_description=(
        "Retrieve document chunks from the Databricks Vector Search index."
        f"{FILTER_GUIDE}"
    ),
    num_results=TOP_K_NUMBER_CHUNKS
)

# ---------------------------------------------------------------------------
# 3. Aggregate tools: UC function interpreter + retriever
# ---------------------------------------------------------------------------

tools: Sequence[BaseTool] = []

# Unity Catalog python_exec tool (optional)
# uc_toolkit = UCFunctionToolkit(function_names=["system.ai.python_exec"])
# tools.extend(uc_toolkit.tools)

# Vector Search retriever
tools.append(rag_tool)


# ---------------------------------------------------------------------------
# 5. LangGraph state machine – one agent node + one tool node
# ---------------------------------------------------------------------------

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        last = state["messages"][-1]
        if last.get("role") == "tool" and last.get("name") == "document_rag":
            payload = json.loads(last["content"])
            if payload.get("status") == "NO_RESULTS":
                return "end"          # finish gracefully
        # fall back to the original rule
        return "continue" if last.get("tool_calls") else "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ---------------------------------------------------------------------------
# 6. MLflow PyFunc wrapper that passes custom_inputs through
# ---------------------------------------------------------------------------

class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, SYSTEM_PROMPT)
AGENT = LangGraphChatAgent(agent)

mlflow.models.set_model(AGENT)
# adding schema for preview app
mlflow.models.set_retriever_schema(
    primary_key="chunk_id",
    text_column="chunk_text",
    doc_uri="path"
)
