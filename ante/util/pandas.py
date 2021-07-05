import pandas as pd
import operator


def is_between_time(series, start_time, end_time, include_start=True, include_end=True):
    start_comparison_operator = operator.ge if include_start else operator.gt
    end_comparison_operator = operator.le if include_end else operator.lt
    greater_than_start = start_comparison_operator(series, start_time)
    lower_than_end = end_comparison_operator(series, end_time)
    return greater_than_start & lower_than_end


def between_time(series, start_time, end_time, include_start=True, include_end=True):
    return series[is_between_time(series, start_time, end_time, include_start, include_end)]


def sliding_window(df, on, window_size, offset, include_start=True, include_end=False):
    date_column = df[on]
    window_size_timedelta = pd.Timedelta(window_size)
    offset_timedelta = pd.Timedelta(offset)
    i = date_column.min()
    end = date_column.max() - window_size_timedelta + offset_timedelta
    while i <= end:
        window_end = i + window_size_timedelta
        window_data = df.loc[is_between_time(date_column, i, window_end, include_start, include_end)]
        i = i + offset_timedelta
        yield i, window_end, window_data
