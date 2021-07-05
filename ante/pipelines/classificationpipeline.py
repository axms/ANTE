from ante.pipelines.pipeline import Pipeline
from autosklearn.classification import AutoSklearnClassifier


class ClassificationPipeline(Pipeline):
    """
    Classification pipeline
    """
    def __init__(self, **pipeline_constructor_params):
        self.estimator = AutoSklearnClassifier(**pipeline_constructor_params)

    def fit(self, x, y):
        self.estimator.fit(x, y)

    def run(self, x):
        return self.estimator.predict(x)

    def as_json(self):
        pipeline = self.estimator.get_models_with_weights()[0][1]
        return pipeline.config.get_dictionary()


class MetaClassificationPipeline(Pipeline):
    """
    Pipeline used to create a classification pipeline
    """
    def __init__(self, time_limit):
        self.time_limit = time_limit

    def run(self, x, y):
        classification_pipeline = ClassificationPipeline(time_left_for_this_task=self.time_limit)
        classification_pipeline.fit(x, y)
        return classification_pipeline

