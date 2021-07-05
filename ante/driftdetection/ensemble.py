import pandas as pd
from ante.util import nested_dict

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score
from skmultiflow.drift_detection import *


class EstimatorBuilder:
    def __init__(self, estimator_class, init_params, name):
        self.estimator_class = estimator_class
        self.init_params = init_params
        self.name = name

    def instantiate(self):
        return self.estimator_class(**self.init_params)


class ClusterMetric:
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def execute(self, params):
        return self.func(*params)


def all_classifiers_options():
    return [
        EstimatorBuilder(
            KMeans,
            {
                "n_clusters": 2
            },
            'KMeans'
        )
    ]


def all_cluster_metrics():
    return [
        ClusterMetric(accuracy_score, "accuracy_score"),
        ClusterMetric(homogeneity_score, "homogeneity_score"),
        ClusterMetric(completeness_score, "completeness_score"),
        ClusterMetric(v_measure_score, "v_measure_score"),
        ClusterMetric(adjusted_rand_score, "adjusted_rand_score"),
        ClusterMetric(adjusted_mutual_info_score, "adjusted_mutual_info_score"),
    ]


def all_detector_options():
    return [
        EstimatorBuilder(ADWIN, {}, 'ADWIN'),
        EstimatorBuilder(DDM, {}, 'DDM'),
        EstimatorBuilder(EDDM, {}, 'EDDM'),
        EstimatorBuilder(HDDM_A, {}, 'HDDM_A'),
        EstimatorBuilder(HDDM_W, {}, 'HDDM_W'),
        EstimatorBuilder(KSWIN, {}, 'KSWIN'),
        EstimatorBuilder(PageHinkley, {}, 'PageHinkley'),
    ]


class ClusterMetricsEnsembleDriftDetector:

    def __init__(self, estimators=all_classifiers_options(), metrics=all_cluster_metrics(), drift_detectors=all_detector_options()):
        self.estimators = estimators
        self.metrics = metrics
        self.drift_detectors = drift_detectors
        self._drift_detector_instances = nested_dict()
        self.reference_concept = []

    def set_reference_concept(self, concept):
        self.reference_concept = pd.DataFrame(concept)
        self.reference_concept["concept"] = 0
        self._drift_detector_instances = nested_dict()
        for clf_option in self.estimators:
            for metric_option in self.metrics:
                for drift_detector_option in self.drift_detectors:
                    self._drift_detector_instances[clf_option.name][metric_option.name][
                        drift_detector_option.name] = drift_detector_option.instantiate()

    def fit_predict(self, concepts):
        concept_predictions = []
        for concept in concepts:
            concept_prediction = []
            test_concept = concept.copy()
            test_concept["concept"] = 1
            training_data = pd.concat([self.reference_concept, test_concept])
            X, y = training_data.drop(["concept"], axis=1), training_data["concept"]
            for clf_option in self.estimators:
                clf = clf_option.instantiate()
                y_hat = clf.fit_predict(X)
                for metric_option in self.metrics:
                    metric_value = metric_option.execute((y, y_hat))
                    for drift_detector_option in self.drift_detectors:
                        detector = self._drift_detector_instances[clf_option.name][metric_option.name][
                            drift_detector_option.name]
                        detector.add_element(metric_value)
                        concept_prediction.append({
                            "classifier": clf_option.name,
                            "metric": metric_option.name,
                            "detector": drift_detector_option.name,
                            "prediction": int(detector.detected_change())
                        })
            concept_predictions.append(pd.DataFrame.from_records(concept_prediction))
        predictions_df = pd.concat(concept_predictions)
        consesus = 1 if 1 in predictions_df["prediction"].values else 0
        return consesus, predictions_df
