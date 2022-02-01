from dl_evaluation_framework.classification_evaluator import (
    DecisionTreeClassificationEvaluator,
    ClassificationEvaluator,
    EvaluationResult,
)
from dl_evaluation_framework.evaluation_manager import EvaluationManager


def test_parse_train_test():
    tt = ClassificationEvaluator.parse_train_test("./tests/classification_directory")
    assert tt is not None
    assert tt.train is not None
    assert tt.test is not None
    assert len(tt.train) > 0
    assert len(tt.test) > 0


def test_decision_tree_classifier():
    evaluator = DecisionTreeClassificationEvaluator()
    vectors = EvaluationManager.read_vector_txt_file(
        "./tests/classification_vectors.txt"
    )
    r = evaluator.evaluate(
        data_directory="./tests/classification_directory",
        vectors=vectors,
    )
    assert r.accuracy == 1.0
    assert "stuttgart" in r.missed


def test_number_missed():
    r = EvaluationResult(
        accuracy=1.0,
        missed={"A", "B", "C"},
        classifier_name="C",
        data_directory="DD",
        gs_size=100,
    )
    assert r.number_missed == 3
    assert r.accuracy == 1.0
    assert r.classifier_name == "C"
    assert r.data_directory == "DD"
    assert r.gs_size == 100
