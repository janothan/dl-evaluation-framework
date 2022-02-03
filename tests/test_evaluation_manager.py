from pathlib import Path

import pandas as pd

from dl_evaluation_framework.evaluation_manager import EvaluationManager, VectorTuple
import shutil

TEST_RESULTS_DIR_STR = "./test_results"
RESULTS_DIR_EXISTS_STR = "./results_dir_exists"
INTERESTED_NODES_STR = "./uris_of_interest.txt"
REDUCED_FILE_STR = "./reduced_vectors.txt"


def test_results_dir_exists():
    rdir = Path(RESULTS_DIR_EXISTS_STR)
    if not rdir.exists():
        rdir.mkdir()
    evaluator = EvaluationManager(test_directory="")
    evaluator.evaluate(
        vector_names_and_files=[], result_directory=RESULTS_DIR_EXISTS_STR
    )


def test_query_dir_does_not_exist():
    evaluator = EvaluationManager(test_directory="DOES-NOT-EXIST")
    evaluator.evaluate(
        vector_names_and_files=[], result_directory="does-not-exist-either"
    )


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
    blank_lines = 0
    with open(REDUCED_FILE_STR, "r", encoding="utf-8") as rfile:
        for line in rfile:
            if line == "\n" or line.strip() == "":
                blank_lines += 1
    assert blank_lines <= 1
    vects = EvaluationManager.read_vector_txt_file(REDUCED_FILE_STR)
    assert len(vects) == 2
    assert "world" in vects
    assert "europa" in vects
    assert vects["europa"][0] == 1

    # some negative tests
    invalid_path_str = "./do-not-write.txt"
    EvaluationManager.reduce_vectors(
        original_vector_file="",
        reduced_vector_file_to_write=invalid_path_str,
        entities_of_interest="./tests/interest2.txt",
    )
    assert Path(invalid_path_str).exists() is False
    EvaluationManager.reduce_vectors(
        original_vector_file="./tests/test_vectors2.txt",
        reduced_vector_file_to_write=invalid_path_str,
        entities_of_interest="",
    )
    assert Path(invalid_path_str).exists() is False


def test_evaluate():
    em = EvaluationManager(test_directory="./tests/result_for_testing")
    em.evaluate(
        vector_names_and_files=[
            VectorTuple(
                vector_path="./tests/test_files/classic_sg.txt",
                vector_name="SG 200 Classic",
            ),
            VectorTuple(
                vector_path="./tests/test_files/classic_sg_oa.txt",
                vector_name="OA SG 200 Classic",
            ),
        ],
        result_directory=TEST_RESULTS_DIR_STR,
    )

    # make sure files were written
    individual_results_path = Path(TEST_RESULTS_DIR_STR).joinpath(
        "individual_results.csv"
    )
    assert individual_results_path.exists()
    individual_df = pd.read_csv(filepath_or_buffer=individual_results_path)
    rows, cols = individual_df.shape
    assert rows > 0
    assert cols > 0

    tcc_results_path = Path(TEST_RESULTS_DIR_STR).joinpath("tc_collection_results.csv")
    assert tcc_results_path.exists()
    tcc_df = pd.read_csv(filepath_or_buffer=tcc_results_path)
    rows, cols = tcc_df.shape
    assert rows > 0
    assert cols > 0

    tcg_results_path = Path(TEST_RESULTS_DIR_STR).joinpath("tc_group_results.csv")
    assert tcg_results_path.exists()
    tcg_df = pd.read_csv(filepath_or_buffer=tcg_results_path)
    rows, cols = tcg_df.shape
    assert rows > 0
    assert cols > 0


def teardown_module(module):
    dir1 = Path(RESULTS_DIR_EXISTS_STR)
    if dir1.exists():
        dir1.rmdir()

    interested_noes_path = Path(INTERESTED_NODES_STR)
    if interested_noes_path.exists():
        interested_noes_path.unlink()

    reduced_file_path = Path(Path(REDUCED_FILE_STR))
    if reduced_file_path.exists():
        reduced_file_path.unlink()

    test_results_dir_path = Path(TEST_RESULTS_DIR_STR)
    if test_results_dir_path.exists():
        shutil.rmtree(test_results_dir_path)
