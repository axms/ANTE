from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import inspect
import pyspark.sql.functions as F
import pandas as pd

from ante.features import calculate_metrics, add_time_diff_column
from ante.pysparkutil import get_or_create_lldb_session, read_pcap_csv
from ante.features import get_in_time_range


class FeatureStore:
    """
    FeatureStore is responsible for transforming raw network traffic into descriptive features used for classifying
    network behavior. The sheer volume of data associated with network traffic makes it impossible to fit all the
    data in memory. To handle this large volume of data all features are calculated using spark. FeatureStore
    calculates two sets of features. The first set is composed of statical information about flow of packets. The
    second set is composed of graph based metrics.


    Parameters
    ----------
    dataset: str
        A valid string path
    window_size: int
        Size of sliding window in which the packets are grouped
    window_offset: int
        Size of the sliding window step
    """

    def __init__(self, dataset: str, window_size: int, window_offset: int):
        self.dataset = dataset
        self.window_size = window_size
        self.window_offset = window_offset

        self._spark = get_or_create_lldb_session()
        self._spark.sparkContext.setCheckpointDir('spark_cache')
        self._df = read_pcap_csv(self.dataset, self._spark)
        capture_boundaries = self._df.agg(
            F.min("parsed_time").alias("capture_start"),
            F.max("parsed_time").alias("capture_end")
        ).collect()[0]

        self.capture_start = capture_boundaries["capture_start"]
        self.capture_end = capture_boundaries["capture_end"]
        self.pointer = self.capture_start

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """

        Get all class parameter names

        Returns
        -------
         return : list[str]
            Return
        """
        init = cls.__init__
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("MetricsGenerator doesn't accept varargs on their __init__")
        return sorted([p.name for p in parameters])

    def get_params(self) -> Dict[str, any]:
        """

        Get all class parameters

        Returns
        -------
        return : dict[str, any]
            Description
        """
        return {
            key: getattr(self, key, None) for key in self._get_param_names()
        }

    def set_params(self, **params):
        """
        Set class parameters

        Parameters
        ----------
        params: dict
            parameters to set
        """
        if not params:
            return self
        immutable_params = ["dataset"]
        valid_params = self._get_param_names()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f'Invalid parameter {key}. '
                                 'Check the list of available parameters '
                                 'with `generator.get_params().keys()`.')
            elif key in immutable_params:
                raise ValueError(f'"{key}" cant be changed after the generator was initiated')
            else:
                setattr(self, key, value)
        return self

    def skip_n_windows(self, n: int):
        """
        Skip the next n windows

        Parameters
        ----------
        n: int
            Number of windows to skip
        """
        self.pointer = self.pointer + timedelta(seconds=self.window_offset) * n

    def rewind_n_windows(self, n: int):
        """
        Go back n windows

        Parameters
        ----------
        n: int
            Number of windows to go back
        """
        self.pointer = self.pointer - timedelta(seconds=self.window_offset) * n

    def _get_pointer_end(self) -> datetime:
        """

        Get current window end date

        Returns
        -------
        return: datetime
        """
        return self.pointer - timedelta(seconds=self.window_size)

    def _get_n_windows(self, n: int, increment_pointer=True) -> List[Tuple[datetime, datetime]]:
        """
        Get start and end dates for the next n windows

        Parameters
        ----------
        n: int
            Number of windows
        increment_pointer: bool
            Update internal pointer?

        Returns
        -------
        windows: list[tuple[datetime, datetime]]
        """
        windows = []
        pointer = self.pointer
        while len(windows) < n and pointer <= self.capture_end:
            new_pointer = pointer + timedelta(seconds=self.window_offset)
            windows.append((pointer, pointer + timedelta(seconds=self.window_size)))
            pointer = new_pointer
        if increment_pointer:
            self.pointer = pointer

        return windows

    def _get_windows_until(self, end, increment_pointer=True) -> List[Tuple[datetime, datetime]]:
        """
        Get windows until the end date of the last window is more or equal than end

        Parameters
        ----------
        end: datetime
            Upper limit for the end of the last window
        increment_pointer: bool
            Update internal pointer?

        Returns
        -------
        windows: list[tuple[datetime, datetime]]
        """
        windows = []
        pointer = self.pointer
        while pointer < end and pointer <= self.capture_end:
            new_pointer = pointer + timedelta(seconds=self.window_offset)
            windows.append((pointer, pointer + timedelta(seconds=self.window_size)))
            pointer = new_pointer
        if increment_pointer:
            self.pointer = pointer
        return windows

    def _get_windows_in_period(self, start, end) -> List[Tuple[datetime, datetime]]:
        """
        Get windows from start until end ignoring internal pointer

        Parameters
        ----------
        start: datetime
            Lower limit for the start of the first window
        end: datetime
            Upper limit for the end of the last window

        Returns
        -------
        windows: list[tuple[datetime, datetime]]
        """
        windows = []
        pointer = start
        while pointer < end and pointer <= end:
            new_pointer = pointer + timedelta(seconds=self.window_offset)
            windows.append((pointer, pointer + timedelta(seconds=self.window_size)))
            pointer = new_pointer
        return windows

    def n_remaining_samples(self) -> int:
        """
        Calculates how many windows until end of capture

        Returns
        -------
        return: int
        """
        return len(self._get_windows_until(self.capture_end, False))

    def has_more_samples(self) -> bool:
        """
        Check if there are more windows in capture

        Returns
        -------
        return: bool
        """
        return self.n_remaining_samples() > 0

    def set_pointer(self, pointer):
        """
        Set internal pointer. Features are calculated starting for pointer

        Parameters
        ----------
        pointer: datetime
            Date from where the next windows start
        """
        self.pointer = pointer

    def restart(self):
        """ Reset internal pointer to start of capture """
        self.set_pointer(self.capture_start)

    def calculate_metrics(self, windows) -> pd.DataFrame:
        """

        Calculate features for date inside windows

        Parameters
        ----------
        windows: list[tuple[datetime, datetime]]
            List defining start and end date for data window

        Returns
        -------
        return: pd.DataFrame

        """
        result = []
        padding = timedelta(seconds=10)
        min_window_start = min(windows, key=lambda x: x[0])[0] - padding
        max_window_end = max(windows, key=lambda x: x[1])[1] + padding

        df_slice = get_in_time_range(self._df, min_window_start, max_window_end)
        df_slice.cache()

        for window_index, (window_start, windows_end) in enumerate(windows):
            window_df = get_in_time_range(df_slice, window_start, windows_end, include_upper=False)
            window_df = add_time_diff_column(window_df)

            metrics_df = calculate_metrics(window_df)
            metrics_df["window_start"] = window_start.isoformat()
            metrics_df["window_end"] = windows_end.isoformat()

            metrics_df["window_index"] = window_index

            result.append(metrics_df)

        return pd.concat(result, sort=False)

    def next_n_samples(self, batch_size=1, increment_pointer=True) -> pd.DataFrame:
        """

        Parameters
        ----------
        batch_size: int
            Desc
        increment_pointer: bool
            Desc

        Returns
        -------
        return: pd.DataFrame
        """
        windows = self._get_n_windows(batch_size, increment_pointer)
        return self.calculate_metrics(windows)

    def next_sample(self, increment_pointer=True) -> pd.DataFrame:
        """

        Parameters
        ----------
        increment_pointer

        Returns
        -------
        return: pd.DataFrame
        """
        windows = self._get_n_windows(1, increment_pointer)
        return self.calculate_metrics(windows)

    def next_samples_until(self, end, increment_pointer=True) -> pd.DataFrame:
        """

        Parameters
        ----------
        end
        increment_pointer

        Returns
        -------
        return: pd.DataFrame
        """
        windows = self._get_windows_until(end, increment_pointer)
        return self.calculate_metrics(windows)

    def samples_in_period(self, start, end) -> pd.DataFrame:
        """

        Parameters
        ----------
        start
        end

        Returns
        -------
        return: pd.DataFrame
        """
        windows = self._get_windows_in_period(start, end)
        return self.calculate_metrics(windows)

    def remaining_samples(self, increment_pointer=False) -> pd.DataFrame:
        """

        Parameters
        ----------
        increment_pointer

        Returns
        -------
        return: pd.DataFrame
        """
        windows = self.next_samples_until(self.capture_end, increment_pointer)
        return self.calculate_metrics(windows)

    def all_samples(self) -> pd.DataFrame:
        """

        Returns
        -------
        return: pd.DataFrame
        """
        windows = self.samples_in_period(self.capture_start, self.capture_end)
        return self.calculate_metrics(windows)

    def __str__(self) -> str:
        """

        Returns
        -------
        return: pd.DataFrame
        """
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items() if not k.startswith("_"))
        return str(self.__class__.__name__) + "(" + params + ")"
