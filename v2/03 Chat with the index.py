# Databricks notebook source
# MAGIC %md
# MAGIC # Todo
# MAGIC
# MAGIC - [-] chat over the index
# MAGIC - [-] how to extend to function calling
# MAGIC - [-] how to pass the possible values for filtering
# MAGIC - [ ] create a tool for filter creation
# MAGIC - [ ] handle zero docs retrieve
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # lib setup

# COMMAND ----------

# lib

%pip install -U -qqqq mlflow databricks-vectorsearch databricks-langchain databricks-agents uv langgraph==0.3.4
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # env parameter setup

# COMMAND ----------

# parameters (redundant with agent.py)

CATALOG   = "edp_dev"
SCHEMA    = "poli"
SUFFIX    = "test_ted"

TRACK_TBL   = f"{CATALOG}.{SCHEMA}.{SUFFIX}_file_tracking"
CHUNK_TBL   = f"{CATALOG}.{SCHEMA}.{SUFFIX}_vector_chunks"
INDEX_NAME  = f"{CATALOG}.{SCHEMA}.{SUFFIX}_chunks_index"

VECTOR_SEARCH_ENDPOINT = "poli_replica_vector_search"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Define the agent in code
# MAGIC
# MAGIC ## refs:
# MAGIC - https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-tool-calling-agent.html

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC
# MAGIC
# MAGIC # agent.py
# MAGIC """
# MAGIC LangGraph + MLflow chat agent that supports runtime Vector Search filters with
# MAGIC strong validation.  Users can only filter by:
# MAGIC
# MAGIC * category – one of {"Construction", "Housing"}
# MAGIC * year     – integer 1900‒2100
# MAGIC * month    – integer 1‒12
# MAGIC
# MAGIC """
# MAGIC from __future__ import annotations
# MAGIC
# MAGIC from typing import Any, Dict, Generator, Optional, Sequence
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     # UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import (
# MAGIC     ChatAgentState,
# MAGIC     ChatAgentToolNode,
# MAGIC )
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC import json
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm: LanguageModelLike = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC TOP_K_NUMBER_CHUNKS = 42
# MAGIC
# MAGIC CATALOG, SCHEMA, SUFFIX = "edp_dev", "poli", "test_ted"
# MAGIC INDEX_NAME = f"{CATALOG}.{SCHEMA}.{SUFFIX}_chunks_index"
# MAGIC
# MAGIC # ---------------------------------------------------------------------------
# MAGIC # Filter guide (inserted into tool description so the LLM sees valid keys
# MAGIC # extra advanced filters example: https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search?language=Python%C2%A0SDK#filters
# MAGIC # ---------------------------------------------------------------------------
# MAGIC
# MAGIC FILTER_GUIDE_OPERATOR = (
# MAGIC     """
# MAGIC     | Filter operator | Behaviour | Filters JSON example |
# MAGIC     |-----------------|-----------|----------------------|
# MAGIC     | **NOT** | Matches rows where the field value **is not** the given value. | `"filters": [ { "key": "color NOT", "value": "red" } ]` |
# MAGIC     | **<** | Field value **less than** the given value. | `"filters": [ { "key": "month <", "value": 9 } ]` |
# MAGIC     | **<=** | Field value **less than or equal** to the given value. | `"filters": [ { "key": "price <=", "value": 200 } ]` |
# MAGIC     | **>** | Field value **greater than** the given value. | `"filters": [ { "key": "price >", "value": 200 } ]` |
# MAGIC     | **>=** | Field value **greater than or equal** to the given value. | `"filters": [ { "key": "price >=", "value": 200 } ]` |
# MAGIC     | **OR** | Checks if **any** of the sub-columns has a listed value. | `"filters": [ { "key": "color1 OR color2", "value": ["red", "blue"] } ]` |
# MAGIC     | **LIKE** | Token-level match inside strings. | `"filters": [ { "key": "title LIKE", "value": "hello" } ]` |
# MAGIC     | *(no operator)* | Exact match; if the value is a list, acts as “IN (…)”. | `"filters": [ { "key": "id", "value": [10, 30] } ]` |
# MAGIC
# MAGIC     ### Example showing implicit **AND** between filter objects
# MAGIC     ```json
# MAGIC     {
# MAGIC     "query": "bridge maintenance",
# MAGIC     "filters": [
# MAGIC         { "key": "category", "value": "Construction" },
# MAGIC         { "key": "year >=",  "value": 2024 }
# MAGIC     ]
# MAGIC     }
# MAGIC     """
# MAGIC )
# MAGIC
# MAGIC FILTER_GUIDE = (
# MAGIC     f"""
# MAGIC     <available values>
# MAGIC     
# MAGIC     Optional `filters` keys: `category` (Construction | Housing), 
# MAGIC     `year` (1900-2100), `month` (1-12). 
# MAGIC
# MAGIC     <The examples>
# MAGIC     {FILTER_GUIDE_OPERATOR}
# MAGIC     """
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # ---------------------------------------------------------------------------
# MAGIC # Global system prompt (high-level instructions, no detailed schema repeat)
# MAGIC # ---------------------------------------------------------------------------
# MAGIC SYSTEM_PROMPT = (
# MAGIC     "You are a knowledgeable, tool-aware assistant. First determine if a tool is needed; "
# MAGIC     "if so, call **one** tool and wait for its result. Use:\n"
# MAGIC     "- `document_rag` for document retrieval. Create filters to refine the search if necessary. If no relevant documents are retrieved, inform the user and suggest refining their criteria.\n"
# MAGIC     # "- `python_exec` for calculations.\n"
# MAGIC     "If no tool is required, provide a direct and concise answer. Always ensure your responses are accurate and relevant."
# MAGIC )
# MAGIC
# MAGIC # ---------------------------------------------------------------------------
# MAGIC # 2. Build a Vector Search tool with dynamic filters
# MAGIC # ---------------------------------------------------------------------------
# MAGIC
# MAGIC
# MAGIC class SafeVectorSearchRetrieverTool(VectorSearchRetrieverTool):
# MAGIC     """Guarantees a string payload even when no docs are found, and truncates big docs."""
# MAGIC     def _run(self, query: str, filters: Optional[list] = None):
# MAGIC         docs = super()._run(query=query, filters=filters)
# MAGIC
# MAGIC         try:
# MAGIC             # --- 1️⃣  No hits  → schema-safe sentinel  -----------------
# MAGIC             if not docs:
# MAGIC                 return {"status": "NO_RESULTS", "num_results": 0, "docs": []}
# MAGIC                 
# MAGIC
# MAGIC             # --- 2️⃣  Convert documents to lightweight strings ---------
# MAGIC             return {"status": "OK", "num_results": len(docs), "docs": docs}
# MAGIC         
# MAGIC         except Exception as e:
# MAGIC             raise Exception(f"Error: {e}, docs: {docs}")
# MAGIC
# MAGIC
# MAGIC rag_tool = SafeVectorSearchRetrieverTool(
# MAGIC     index_name=INDEX_NAME,
# MAGIC     columns=["chunk_text", "chunk_id", "path"],
# MAGIC     query_type="HYBRID",
# MAGIC     tool_name="document_rag",
# MAGIC     tool_description=(
# MAGIC         "Retrieve document chunks from the Databricks Vector Search index."
# MAGIC         f"{FILTER_GUIDE}"
# MAGIC     ),
# MAGIC     num_results=TOP_K_NUMBER_CHUNKS
# MAGIC )
# MAGIC
# MAGIC # ---------------------------------------------------------------------------
# MAGIC # 3. Aggregate tools: UC function interpreter + retriever
# MAGIC # ---------------------------------------------------------------------------
# MAGIC
# MAGIC tools: Sequence[BaseTool] = []
# MAGIC
# MAGIC # Unity Catalog python_exec tool (optional)
# MAGIC # uc_toolkit = UCFunctionToolkit(function_names=["system.ai.python_exec"])
# MAGIC # tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # Vector Search retriever
# MAGIC tools.append(rag_tool)
# MAGIC
# MAGIC
# MAGIC # ---------------------------------------------------------------------------
# MAGIC # 5. LangGraph state machine – one agent node + one tool node
# MAGIC # ---------------------------------------------------------------------------
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         last = state["messages"][-1]
# MAGIC         if last.get("role") == "tool" and last.get("name") == "document_rag":
# MAGIC             payload = json.loads(last["content"])
# MAGIC             if payload.get("status") == "NO_RESULTS":
# MAGIC                 return "end"          # finish gracefully
# MAGIC         # fall back to the original rule
# MAGIC         return "continue" if last.get("tool_calls") else "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC # ---------------------------------------------------------------------------
# MAGIC # 6. MLflow PyFunc wrapper that passes custom_inputs through
# MAGIC # ---------------------------------------------------------------------------
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC mlflow.langchain.autolog()
# MAGIC agent = create_tool_calling_agent(llm, tools, SYSTEM_PROMPT)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC
# MAGIC mlflow.models.set_model(AGENT)
# MAGIC # adding schema for preview app
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key="chunk_id",
# MAGIC     text_column="chunk_text",
# MAGIC     doc_uri="path"
# MAGIC )
# MAGIC

# COMMAND ----------

from IPython.display import Image, display
from agent import AGENT, agent

display(Image(agent.get_graph().draw_mermaid_png()))


# COMMAND ----------

# MAGIC %md
# MAGIC # Test the agent

# COMMAND ----------

# to reload the agent.py
dbutils.library.restartPython() 

# COMMAND ----------


if 0:
  # test the index search
  from agent import rag_tool

  rag_tool.invoke(
      {
    "query": "Construction or Housing",
    "filters": [
      {
        "key": "month <",
        "value": 9
      }]
      }
  )

# COMMAND ----------


if 1:
    from agent import AGENT

    AGENT.predict(
        {"messages": [
            {"role": "user", "content": "I want to know about Housing related information in 2024 SEP"}
            ]})



# COMMAND ----------


if 1:
    from agent import AGENT


    msg = (
        """
        I want to know about Housing related information before 2024
        """
    )


    AGENT.predict(
        {"messages": [
            {"role": "user", "content": msg}
            ]})



# COMMAND ----------


if 1:
    from agent import AGENT


    msg = (
        """
        I want to know about Housing and Construction related information after 2024
        """
    )


    AGENT.predict(
        {"messages": [
            {"role": "user", "content": msg}
            ]})

