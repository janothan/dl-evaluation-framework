from dl_evaluation_framework.classification_evaluator import (
    DecisionTreeClassificationEvaluator,
    ClassificationEvaluator,
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
        results_file="./some-file.txt",
    )
    assert r.accuracy == 1.0
    assert "stuttgart" in r.missed
