from dl_evaluation_framework.classification_evaluator import DecisionTreeClassificationEvaluator
from dl_evaluation_framework.evaluation_manager import EvaluationManager

def test_evaluate():
    evaluator = DecisionTreeClassificationEvaluator()
    vectors = EvaluationManager.read_vector_txt_file("./tests/classification_vectors.txt")
    #evaluator.evaluate(data_directory="./classification_directory",
    #                   vectors=vectors,
    #                   results_file="./some-file.txt")
