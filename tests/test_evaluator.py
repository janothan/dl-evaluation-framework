from pathlib import Path
from dl_evaluation_framework.evaluator import Evaluator

RESULTS_DIR_EXISTS_STR = "./results_dir_exists"
INTERESTED_NODES_STR  = "./uris_of_interest.txt"

def test_results_dir_exists():
    rdir = Path(RESULTS_DIR_EXISTS_STR)
    if not rdir.exists():
        rdir.mkdir()
    evaluator = Evaluator(test_directory="")
    evaluator.evaluate(vector_file="", result_directory=RESULTS_DIR_EXISTS_STR)


def test_query_dir_does_not_exist():
    evaluator = Evaluator(test_directory="DOES-NOT-EXIST")
    evaluator.evaluate(vector_file="", result_directory="does-not-exist-either")


def test_read_vector_txt_file():
    print("ABC")
    result = Evaluator.read_vector_txt_file(vector_file="./tests/test_vectors.txt")
    assert "hello" in result
    assert result["hello"][0] == 1
    assert result["hello"][1] == 2
    assert result["hello"][2] == -4
    assert result["hello"][3] == 0.5


def test_read_iris_from_file():
    result = Evaluator.read_iris_from_file(file_to_read_from="./tests/sample_iri_file.txt")
    assert len(result) == 7
    assert "http://dbpedia.org/resource/Attica,_Kansas" in result

def test_write_uris_of_interest_to_file():
    evaluator = Evaluator(test_directory="./tests/result")
    evaluator.write_uris_of_interest_to_file(file_to_write=INTERESTED_NODES_STR)
    file_to_write_path = Path(INTERESTED_NODES_STR)
    assert file_to_write_path.exists()


def teardown_module(module):
    dir1 = Path(RESULTS_DIR_EXISTS_STR)
    if dir1.exists():
        dir1.rmdir()
    Path(INTERESTED_NODES_STR).unlink()
