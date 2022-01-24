from pathlib import Path
from dl_evaluation_framework.evaluator import Evaluator

RESULTS_DIR_EXISTS_STR = "./results_dir_exists"


def test_results_dir_exists():
    rdir = Path(RESULTS_DIR_EXISTS_STR)
    if not rdir.exists():
        rdir.mkdir()
    evaluator = Evaluator(query_directory="")
    evaluator.evaluate(vector_file="", result_directory=RESULTS_DIR_EXISTS_STR)


def test_query_dir_does_not_exist():
    evaluator = Evaluator(query_directory="DOES-NOT-EXIST")
    evaluator.evaluate(vector_file="", result_directory="does-not-exist-either")


def teardown_module(module):
    dir1 = Path(RESULTS_DIR_EXISTS_STR)
    if dir1.exists():
        dir1.rmdir()
