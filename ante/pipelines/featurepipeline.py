from pathlib import Path
from datetime import timedelta, datetime
from functools import partial
from prefect import task
from prefect.engine.results import LocalResult
import prefect
import pandas as pd
import numpy as np
from prefect import task, Flow, Parameter, Task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

from ante.features import FeatureStore
from ante.pipelines.pipeline import PrefectFlowPipeline


def timestamp_to_date_string(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d--%H-%M-%S")


def generate_task_run_target_name(**kwargs):
    dataset = Path(kwargs['feature_store'].dataset).stem
    window_size = kwargs['feature_store'].window_size
    window_offset = kwargs['feature_store'].window_offset
    start_date_string = timestamp_to_date_string(kwargs['start_date'])
    end_date_string = timestamp_to_date_string(kwargs['end_date'])
    return f"d_{dataset}_w_{window_size}_o_{window_offset}_s_{start_date_string}_f_{end_date_string}.csv"


@task
def create_feature_store(dataset, window_size, window_offset):
    return FeatureStore(dataset, window_size=window_size, window_offset=window_offset)


@task(
    target=generate_task_run_target_name,
    checkpoint=True,
    result=LocalResult(
        dir="../../data/processed/prefect_results",
        serializer=PandasSerializer(file_type="csv", serialize_kwargs={"index": False})
    )
)
def create_features(feature_store, start_date, end_date):
    feature_store.set_pointer(datetime.fromtimestamp(start_date))
    return feature_store.next_samples_until(datetime.fromtimestamp(end_date))


@task
def add_label_column(df, bots):
    df["isBot"] = df["IP"].isin(bots).astype(int)
    return df


def build_feature_pipeline_flow():
    with Flow("feature-pipeline") as feature_pipeline:
        dataset = Parameter("dataset")

        start_date = Parameter("start_date")
        end_date = Parameter("end_date")
        window_size = Parameter("window_size")
        window_offset = Parameter("window_offset")
        bots = Parameter("bots")

        feature_store = create_feature_store(dataset, window_size, window_offset)

        df = create_features(feature_store, start_date, end_date)

        add_label_column(df, bots, task_args=dict(slug="output"))
    return feature_pipeline


FeaturesPipeline = partial(PrefectFlowPipeline, flow_factory=build_feature_pipeline_flow)
