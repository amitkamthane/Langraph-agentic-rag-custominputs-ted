# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # TODo and notes
# MAGIC
# MAGIC
# MAGIC - integrate with a mlflow tracking
# MAGIC - need to extract meta information, either from user provide (currently grab it from diretory) or I need build a extractor, extract from the pdf file
# MAGIC
# MAGIC Steps
# MAGIC - scan volume for file path

# COMMAND ----------

# MAGIC %md
# MAGIC # Libs

# COMMAND ----------

# MAGIC
# MAGIC %pip install -U delta-spark                                      # Delta APIs
# MAGIC # # optional: use Auto Loader & binaryFile without extra libs
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# COMMAND ----------
CATALOG = "edp_dev"                      # Unity Catalog catalog
SCHEMA  = "poli"                         # schema
VOLUME  = "test_sandpit_ted/poli_replica/source_docs"   # volume path under schema

ROOT    = f"dbfs:/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/"  # must end with "/"

SUFFIX = "test_ted"
TRACK_TBL = f"{CATALOG}.{SCHEMA}.{SUFFIX}_file_tracking"
FILE_TYPES = "*.[pP][dD][fF]"            # .pdf & .PDF

OVERRIDE = False


# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration env varibles etc
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # helper functions

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # worklflow
# MAGIC

# COMMAND ----------

# copy file to my sandpit

if 0:
    src = "dbfs:/Volumes/edp_dev/poli/source_docs"
    dst = "dbfs:/Volumes/edp_dev/poli/test_sandpit_ted/poli_replica/source_docs"
    dbutils.fs.cp(src, dst, recurse=True)      # deep copy


# COMMAND ----------

if OVERRIDE:
    spark.sql(f"DROP TABLE IF EXISTS {TRACK_TBL}")

# COMMAND ----------

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


files_df = (spark.read.format("binaryFile")                        # binary source
            .option("recursiveFileLookup", "true")                 # walk sub-dirs
            .option("pathGlobFilter", FILE_TYPES)                  # case-insensitive glob
            .load(ROOT)
            .select(
                F.input_file_name().alias("path"),
                F.col("length"),
                F.col("modificationTime").alias("modification_ts"))
            .withColumn("ingest_ts", F.current_timestamp()))

display(files_df)


# COMMAND ----------

# COMMAND ----------
rel_df = files_df.withColumn(
    "rel_path", F.regexp_replace("path", F.lit(ROOT), "")
).withColumn(  # drop fixed prefix
    "levels", F.split("rel_path", "/")
)

parsed_df = (
    rel_df.withColumn("category", F.col("levels").getItem(0))
    .withColumn("year", F.col("levels").getItem(1).cast("int"))
    .withColumn("month", F.col("levels").getItem(2))
)

# we need to create a mapping, for the month column, I need to parse the natural text word into a month number in int
month_mapping = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

parse_month_udf = F.udf(
    lambda month: month_mapping.get(month.lower(), None), IntegerType()
)

parsed_df = parsed_df.withColumn("month", parse_month_udf(F.col("month")))


display(parsed_df)

# COMMAND ----------

# COMMAND ----------
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TRACK_TBL} (
  path            STRING,
  length          BIGINT,
  modification_ts TIMESTAMP,
  ingest_ts       TIMESTAMP,
  category        STRING,
  year            INT,
  month           STRING,
  status          STRING,      -- NEW | UPDATED | DELETED | INDEXED
  version         BIGINT,
  is_current      BOOLEAN,
  indexed_at      TIMESTAMP
) USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""
)


# COMMAND ----------

# COMMAND ----------
from delta.tables import DeltaTable
track = DeltaTable.forName(spark, TRACK_TBL)

(track.alias("t")
 .merge(parsed_df.alias("s"), "t.path = s.path")
 .whenMatchedUpdate(
      condition="s.modification_ts > t.modification_ts",
      set = { "length":         "s.length",
              "modification_ts":"s.modification_ts",
              "ingest_ts":      "s.ingest_ts",
              "category":       "s.category",
              "year":           "s.year",
              "month":          "s.month",
              "status":         "'UPDATED'",
              "version":        "t.version + 1",
              "is_current":     "true"})
 .whenNotMatchedInsert(values = {
              "path":           "s.path",
              "length":         "s.length",
              "modification_ts":"s.modification_ts",
              "ingest_ts":      "s.ingest_ts",
              "category":       "s.category",
              "year":           "s.year",
              "month":          "s.month",
              "status":         "'NEW'",
              "version":        "1",
              "is_current":     "true"})
 .execute())


# COMMAND ----------

# COMMAND ----------
live_paths = [r.path for r in parsed_df.select("path").collect()]

track.update(
    condition=f"is_current = true AND path NOT IN ({','.join([f'\"{p}\"' for p in live_paths])})",
    set={"status": "'DELETED'", "is_current": "false"}
)

# COMMAND ----------

# COMMAND ----------
display(spark.sql(f"SELECT status, COUNT(*) FROM {TRACK_TBL} GROUP BY status"))


# COMMAND ----------

display(spark.sql(f"SELECT * FROM {TRACK_TBL}"))

# COMMAND ----------


display(
    spark.sql(
    f"SELECT category, status, count(*) FROM {TRACK_TBL} GROUP BY category, status")
        )