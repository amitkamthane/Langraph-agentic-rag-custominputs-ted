# Databricks notebook source
# lib

%pip install -U -qqqq mlflow databricks-vectorsearch databricks-langchain databricks-agents uv langgraph==0.3.4
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Log the agent as an MLflow model
# MAGIC

# COMMAND ----------


import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the model to Unity Catalog
# MAGIC

# COMMAND ----------


CATALOG   = "edp_dev"
SCHEMA    = "poli"
SUFFIX    = "test_ted"

# TODO: define the catalog, schema, and model name for your UC model
model_name = "agentic_rag_poli"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{SUFFIX}_{model_name}"

mlflow.set_registry_uri("databricks-uc")

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)


# COMMAND ----------

# MAGIC %md
# MAGIC # DEPLOY

# COMMAND ----------


from databricks import agents

deployment = agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version)

# Retrieve the query endpoint URL for making API requests
deployment.query_endpoint

