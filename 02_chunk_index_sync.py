# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 02 – Chunk, Embed, Index, and Sync *(Spark & MLflow)*
# MAGIC

# COMMAND ----------

# %pip install -qqqq -U pypdf==4.1.0 databricks-vectorsearch \
#             langchain-text-splitters==0.2.2 mlflow mlflow-skinny

# COMMAND ----------

import os, uuid
from urllib.parse import unquote

import mlflow
from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import pandas_udf, PandasUDFType
from delta.tables import DeltaTable

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoTokenizer, AutoModel
import torch

# --- PATCH helper ---------------------------------------------------------

def to_local_dbfs(path: str) -> str:
    """Convert dbfs:/ URI → /Volumes/ and decode spaces (%20)."""
    local = unquote(path.replace("dbfs:/", "/"))
    if not os.path.isfile(local):
        raise FileNotFoundError(local)
    return local

# COMMAND ----------

# ╔════════════════════════════════════╗
# ║            CONFIG (from 00)        ║
# ╚════════════════════════════════════╝
CATALOG   = "edp_dev"
SCHEMA    = "poli"
SUFFIX    = "test_ted"

TRACK_TBL   = f"{CATALOG}.{SCHEMA}.{SUFFIX}_file_tracking"
CHUNK_TBL   = f"{CATALOG}.{SCHEMA}.{SUFFIX}_vector_chunks"
INDEX_NAME  = f"{CATALOG}.{SCHEMA}.{SUFFIX}_chunks_index"

CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 256

VECTOR_SEARCH_ENDPOINT = "poli_replica_vector_search"
EMBEDDING_MODEL_ENDPOINT_NAME = "databricks-gte-large-en"
PIPELINE_TYPE = "TRIGGERED"

OVERRIDE = False

# COMMAND ----------

# Util: PDF → text

def pdf_to_text(path: str) -> str:
    # convert dbfs:/... to /Volumes/... for local access
    local_path = to_local_dbfs(path)
    reader = PdfReader(local_path)
    return "\n".join(pg.extract_text() or "" for pg in reader.pages)

# COMMAND ----------

# Broadcast heavy objs

_splitter_bc  = spark.sparkContext.broadcast(
    RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
)

@pandas_udf(
    "path string, chunk_id string, chunk_text string, category string, year int, month int",
    PandasUDFType.GROUPED_MAP,
)
def chunk_file(pdf_batch):
    """One PDF → many chunk rows (runs in worker)."""
    import pandas as pd, uuid
    path      = pdf_batch.loc[0, "path"]
    category  = pdf_batch.loc[0, "category"]
    year      = int(pdf_batch.loc[0, "year"])
    month     = int(pdf_batch.loc[0, "month"])

    raw_text = pdf_to_text(path)  # uses patched converter
    chunks   = _splitter_bc.value.split_text(raw_text)

    rows = [{
        "path": path,           # keep original URI for lineage
        "chunk_id": f"{uuid.uuid4().hex}_{i}",
        "chunk_text": c,
        "category": category,
        "year": year,
        "month": month
    } for i, c in enumerate(chunks)]
    return pd.DataFrame(rows)

# COMMAND ----------

display(ft)

# COMMAND ----------

mlflow.set_experiment("/Shared/poli_vector_pipeline")

with mlflow.start_run(run_name="02_chunk_index_sync") as run:

    mlflow.log_params({"model": EMBEDDING_MODEL_ENDPOINT_NAME,
                       "chunk_size": CHUNK_SIZE,
                       "chunk_overlap": CHUNK_OVERLAP})

    ft = spark.read.table(TRACK_TBL)

    status_for_chunk_and_index = ["NEW", "UPDATED"]

    if OVERRIDE:
        spark.sql(f"DROP TABLE IF EXISTS {CHUNK_TBL}")
        status_for_chunk_and_index.append("INDEXED")

    to_index = ft.filter(F.col("status").isin(status_for_chunk_and_index))
    to_delete= ft.filter(F.col("status") == "DELETED")

    mlflow.log_param("rows_to_index", to_index.count())
    mlflow.log_param("rows_to_delete", to_delete.count())

    # CHUNK UPDATE --------------------------------------------------
    if to_index.count():

        chunks_df = (to_index
                      .groupBy("category")
                      .apply(chunk_file))

        if not spark._jsparkSession.catalog().tableExists(CHUNK_TBL):
            # First‑time creation with Change Data Feed enabled
            (chunks_df.write
                    .format("delta")
                    .mode("overwrite")
                    .option("mergeSchema", "true")
                    .option("delta.enableChangeDataFeed", "true")  # 🔑 CDF on
                    .saveAsTable(CHUNK_TBL))       
        else:
            # Ensure CDF property stays on (in case table was created manually)
            spark.sql(f"ALTER TABLE {CHUNK_TBL} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

            tgt = DeltaTable.forName(spark, CHUNK_TBL)
            (tgt.alias("t")
                .merge(chunks_df.alias("s"), "t.chunk_id = s.chunk_id")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute())

    # ─ Delete --------------------------------------------------------
    if to_delete.count():
        # dataframe‑API deletion: build distinct path DF and use whenMatchedDelete()
        del_paths = to_delete.select("path").distinct()
        tgt = DeltaTable.forName(spark, CHUNK_TBL)
        (tgt.alias("t")
                .merge(del_paths.alias("d"), "t.path = d.path")
                .whenMatchedDelete()
                .execute())

    # VECTOR SEARCH INDEX chunk table ------------------------------
    vsc = VectorSearchClient()

    def find_index(endpoint_name, index_name):
        all_indexes = vsc.list_indexes(name=endpoint_name).get("vector_indexes", [])
        return index_name in map(lambda i: i.get("name"), all_indexes)

    if OVERRIDE or not find_index(VECTOR_SEARCH_ENDPOINT, INDEX_NAME):

        if OVERRIDE:
            try:
                resp = vsc.delete_index(
                    endpoint_name=VECTOR_SEARCH_ENDPOINT,
                    index_name=INDEX_NAME
                )
            except Exception as e:
                print(f"Vector search deletion status: \n {e}")

        index = vsc.create_delta_sync_index_and_wait(
                index_name   = INDEX_NAME,
                source_table_name   = CHUNK_TBL,   
                primary_key  = "chunk_id",
                endpoint_name = VECTOR_SEARCH_ENDPOINT,
                embedding_model_endpoint_name = EMBEDDING_MODEL_ENDPOINT_NAME,
                embedding_source_column = "chunk_text",
                pipeline_type = PIPELINE_TYPE
            )
    else:
        index = vsc.get_index(VECTOR_SEARCH_ENDPOINT, INDEX_NAME)
        # mannual sync over the chunk table
        index.sync()

    # mark as INDEXED
    upd = to_index.select("path") \
            .withColumn("status", F.lit("INDEXED")) \
            .withColumn("indexed_at", F.current_timestamp())

    DeltaTable.forName(spark, TRACK_TBL) \
        .alias("t").merge(upd.alias("u"), "t.path = u.path") \
        .whenMatchedUpdate(set={"status": "u.status", "indexed_at": "u.indexed_at"}) \
        .execute()

    mlflow.log_metric("vector_rows", spark.read.table(CHUNK_TBL).count())
    mlflow.end_run()

# COMMAND ----------

index.similarity_search(columns=["chunk_text", "chunk_id", "path", "year", "month"], query_text="Who are the attendess please list them")


# COMMAND ----------

display(spark.sql(f"SELECT status, count(*) FROM {TRACK_TBL} GROUP BY status"))

# COMMAND ----------

display(spark.sql(f"SELECT category, year, month, count(*) FROM {CHUNK_TBL} GROUP BY category, year, month"))

# COMMAND ----------

