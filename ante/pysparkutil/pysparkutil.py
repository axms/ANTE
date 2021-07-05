from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType, TimestampType
import pyspark.sql.functions as F


def get_or_create_lldb_session():
    spark_session = SparkSession \
        .builder \
        .appName("lldb") \
        .getOrCreate()
    spark_session.sparkContext.setLogLevel('ERROR')
    return spark_session


def build_csv_schema():
    return StructType([
        StructField("frame.time", StringType()),
        StructField("frame.time_epoch", DoubleType()),
        StructField("_ws.col.Source", StringType()),
        StructField("_ws.col.Destination", StringType()),
        StructField("_ws.col.Protocol", StringType()),
        StructField("frame.len", IntegerType())
    ])


def read_pcap_csv(filename, spark_session):
    schema = build_csv_schema()
    df = spark_session.read.csv(
        filename, header=True, sep=";", schema=schema
    )
    df = df \
        .withColumnRenamed("frame.time", "time") \
        .withColumnRenamed("frame.time_epoch", "time_epoch") \
        .withColumnRenamed("_ws.col.Source", "Source") \
        .withColumnRenamed("_ws.col.Destination", "Destination") \
        .withColumnRenamed("_ws.col.Protocol", "Protocol") \
        .withColumnRenamed("frame.len", "frame_len")
    df = df.withColumn('parsed_time', F.from_utc_timestamp(df['time_epoch'].cast(TimestampType()), 'UTC'))
    df = df.withColumn("id", F.monotonically_increasing_id())
    return df.orderBy("parsed_time")
