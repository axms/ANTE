from abc import ABCMeta, abstractmethod
from ante.util import get_first_or_none


class Pipeline(metaclass=ABCMeta):
    """
    Base class for all pipelines
    """
    @abstractmethod
    def run(self, **kwargs):
        pass


class PrefectFlowPipeline(Pipeline):
    """
    Pipeline with a prefect flow backend
    """
    def __init__(self, flow_factory):
        self.prefect_flow = flow_factory()
        self.execution_state = None

    def _extract_output(self):
        output_task = get_first_or_none(self.prefect_flow.get_tasks(slug="output"))
        output_task_result = self.execution_state.result[output_task].result
        return output_task_result

    def run(self, **kwargs):
        self.execution_state = self.prefect_flow.run(**kwargs)
        return self._extract_output()
