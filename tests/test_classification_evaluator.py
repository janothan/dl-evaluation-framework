from dl_evaluation_framework.classification_evaluator import DecisionTreeClassificationEvaluator, \
    ClassificationEvaluator
from dl_evaluation_framework.evaluation_manager import EvaluationManager


def test_train_test_file_to_df():
    tt = ClassificationEvaluator.parse_train_test("./tests/classification_directory")
    assert tt is not None
    assert tt.train is not None
    assert tt.test is not None
    assert len(tt.train) > 0
    assert len(tt.test) > 0

def test_parse_train_test():
    cdir = ClassificationEvaluator.parse_train_test("./tests/classification_vectors.txt")

def later():
    vectors = EvaluationManager.read_vector_txt_file("./tests/classification_vectors.txt")

