from pathlib import Path
from dl_evaluation_framework.evaluation_manager import EvaluationManager

RESULTS_DIR_EXISTS_STR = "./results_dir_exists"
INTERESTED_NODES_STR = "./uris_of_interest.txt"
REDUCED_FILE_STR = "./reduced_vectors.txt"


def test_results_dir_exists():
    rdir = Path(RESULTS_DIR_EXISTS_STR)
    if not rdir.exists():
        rdir.mkdir()
    evaluator = EvaluationManager(test_directory="")
    evaluator.evaluate(vector_files="", result_directory=RESULTS_DIR_EXISTS_STR)


def test_query_dir_does_not_exist():
    evaluator = EvaluationManager(test_directory="DOES-NOT-EXIST")
    evaluator.evaluate(vector_files="", result_directory="does-not-exist-either")


def test_read_vector_txt_file():
    print("ABC")
    result = EvaluationManager.read_vector_txt_file(
        vector_file="./tests/test_vectors.txt"
    )
    assert "hello" in result
    assert result["hello"][0] == 1
    assert result["hello"][1] == 2
    assert result["hello"][2] == -4
    assert result["hello"][3] == 0.5


def test_read_iris_from_file():
    result = EvaluationManager.read_iris_from_file(
        file_to_read_from="./tests/sample_iri_file.txt"
    )
    assert len(result) == 7
    assert "http://dbpedia.org/resource/Attica,_Kansas" in result


def test_write_uris_of_interest_to_file():
    evaluator = EvaluationManager(test_directory="./tests/result")
    evaluator.write_uris_of_interest_to_file(file_to_write=INTERESTED_NODES_STR)
    file_to_write_path = Path(INTERESTED_NODES_STR)
    assert file_to_write_path.exists()


def test_reduce_vectors():
    EvaluationManager.reduce_vectors(
        original_vector_file="./tests/test_vectors2.txt",
        reduced_vector_file_to_write=REDUCED_FILE_STR,
        entities_of_interest="./tests/interest2.txt",
    )
    assert Path(REDUCED_FILE_STR).exists()


def teardown_module(module):
    dir1 = Path(RESULTS_DIR_EXISTS_STR)
    if dir1.exists():
        dir1.rmdir()
    Path(INTERESTED_NODES_STR).unlink()
    Path(REDUCED_FILE_STR).unlink()
