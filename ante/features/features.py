
from typing import List, Dict, Tuple
import pandas as pd
import networkx as nx
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.group import GroupedData
from pyspark.sql.window import Window
from pyspark.sql import DataFrame as SparkDataFrame


def get_in_time_range(df, start, end, include_lower=True, include_upper=True):
    mask = None
    if start:
        start_with_timezone = start.astimezone(pytz.utc)
        lower_bound_comparison = col("parsed_time") >= start_with_timezone if include_lower else col(
            "parsed_time") > start_with_timezone
        mask = lower_bound_comparison
    if end:
        end_with_timezone = end.astimezone(pytz.utc)
        upper_bound_comparison = col("parsed_time") <= end_with_timezone if include_upper else col(
            "parsed_time") < end_with_timezone
        mask = upper_bound_comparison if mask is None else (mask & upper_bound_comparison)
    if mask is None:
        return df
    return df.filter(mask)


def build_graph_from_df(df: pd.DataFrame) -> nx.DiGraph:
    """

    Parameters
    ----------
    df: pd.DataFrame
        Description

    Returns
    -------
    return: nx.DiGraph
    """
    graph_data = df.groupBy(
        col("Source"),
        col("Destination")
    ).count().toPandas()
    return nx.from_pandas_edgelist(graph_data, 'Source', 'Destination', ['count'], create_using=nx.DiGraph)


def group_by_source(df: SparkDataFrame) -> GroupedData:
    """

    Parameters
    ----------
    df: pd.Dataframe
        Description
    Returns
    -------
    return: GroupedData
    """
    return df.groupby(col("Source").alias("IP"))


def group_by_destination(df: SparkDataFrame) -> GroupedData:
    """

    Parameters
    ----------
    df: pd.Dataframe
        Description
    Returns
    -------
    return: GroupedData
    """
    return df.groupby(col("Destination").alias("IP"))


def eigenvector_centrality_or_empty(graph: nx.DiGraph) -> Dict[str, float]:
    """

    Parameters
    ----------
    graph: nx.DiGraph

    Returns
    -------
    return: dict[str, float]
    """
    try:
        return nx.eigenvector_centrality(graph)
    except Exception:
        return dict()


def calculate_graph_metrics(graph: nx.DiGraph) -> pd.DataFrame:
    """

    Parameters
    ----------
    graph: nx.DiGraph

    Returns
    -------
    return:  pd.DataFrame
    """
    def fill_node_data(non_filled_data: Dict[str, float]):
        return {**{node: 0 for node in all_nodes}, **non_filled_data}

    all_nodes = list(graph.nodes())
    node_data = {
        'in_degree': fill_node_data(dict(graph.in_degree())),
        'out_degree': fill_node_data(dict(graph.out_degree())),
        'centrality': fill_node_data(nx.betweenness_centrality(graph)),
        'eigenvector_centrality': fill_node_data(eigenvector_centrality_or_empty(graph))
    }
    return pd.DataFrame.from_dict(node_data).rename_axis('IP').reset_index().fillna(0)


def add_time_diff_column(df: SparkDataFrame) -> SparkDataFrame:
    """

    Parameters
    ----------
    df: SparkDataFrame

    Returns
    -------
    return: SparkDataFrame
    """
    def build_time_lag_column(field):
        return F.lag(col('time_epoch')).over(Window.partitionBy(field).orderBy("time_epoch"))

    def add_time_lag_column(df_without_column, field):
        return df_without_column.withColumn('time_epoch_prev', build_time_lag_column(field)) \
            .withColumn(f'{field}_time_diff', (col("time_epoch") - col('time_epoch_prev'))) \
            .drop('time_epoch_prev')

    with_source_lag = add_time_lag_column(df, "Source")
    return add_time_lag_column(with_source_lag, "Destination")


def get_metrics_by_source(df: SparkDataFrame) -> SparkDataFrame:
    """

    Parameters
    ----------
    df: SparkDataFrame

    Returns
    -------
    return: SparkDataFrame
    """
    frame_len = col("frame_len")
    time_diff = col("Source_time_diff")
    metrics = [
        F.count("*").alias("Spc"),
        F.sum(frame_len).alias("Tss"),
        F.min(frame_len).alias("Smin"),
        F.max(frame_len).alias("Smax"),
        F.mean(frame_len).alias("Savg"),
        F.stddev(frame_len).alias("Svar"),
        F.min(time_diff).alias("SITmin"),
        F.max(time_diff).alias("SITmax"),
        F.mean(time_diff).alias("SITavg"),
        F.stddev(time_diff).alias("SITvar"),
    ]
    grouped_by_source = group_by_source(df)
    return grouped_by_source.agg(*metrics)


def get_metrics_by_destination(df: SparkDataFrame) -> SparkDataFrame:
    """

    Parameters
    ----------
    df: SparkDataFrame

    Returns
    -------
    return: SparkDataFrame
    """
    frame_len = col("frame_len")
    time_diff = col("Destination_time_diff")
    metrics = [
        F.count("*").alias("Rpc"),
        F.sum(frame_len).alias("Tsr"),
        F.mean(frame_len).alias("Ravg"),
        F.stddev(frame_len).alias("Rvar"),
        F.min(time_diff).alias("RITmin"),
        F.max(time_diff).alias("RITmax"),
        F.mean(time_diff).alias("RITavg"),
        F.stddev(time_diff).alias("RITvar"),
    ]
    grouped_by_destination = group_by_destination(df)
    return grouped_by_destination.agg(*metrics)


def add_label_column(metrics_df: pd.DataFrame, bots: List[str]):
    """

    Parameters
    ----------
    metrics_df: pd.DataFrame
    bots: List[str]

    """
    metrics_df["isBot"] = metrics_df["IP"].isin(bots).astype(int)


def calculate_metrics(df_slice: SparkDataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    df_slice: SparkDataFrame

    Returns
    -------
    return: pd.DataFrame
    """
    by_source_metrics = get_metrics_by_source(df_slice)
    by_destination_metrics = get_metrics_by_destination(df_slice)

    by_source_df = by_source_metrics.toPandas()

    by_destination = by_destination_metrics.toPandas()

    graph = build_graph_from_df(df_slice)

    graph_metrics_df = calculate_graph_metrics(graph)

    metrics_df = pd.merge(by_source_df, by_destination, how="outer", on="IP").fillna(0)
    metrics_df = pd.merge(metrics_df, graph_metrics_df, how="outer", on="IP")

    return metrics_df


