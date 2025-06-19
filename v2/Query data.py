# Databricks notebook source
CATALOG   = "edp_dev"
SCHEMA    = "poli"
SUFFIX    = "test_ted"

TRACK_TBL   = f"{CATALOG}.{SCHEMA}.{SUFFIX}_file_tracking"

# COMMAND ----------


display(
    spark.sql(
    f"SELECT category, status, count(*) FROM {TRACK_TBL} GROUP BY category, status")
        )