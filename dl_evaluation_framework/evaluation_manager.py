import logging.config
from pathlib import Path
from dl_evaluation_framework.classification_evaluator import (
    ClassificationEvaluator,
    DecisionTreeClassificationEvaluator,
)

# some logging configuration
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

logconf_file = Path.joinpath(Path(__file__).parent.resolve(), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class EvaluationManager:
    """Highest coordination layer for the evaluation process."""

    def __init__(self, test_directory: str):
        """Constructor

        Parameters
        ----------
        test_directory : str
            The path to the query results directory (containing the URIs in separate files).
        """
        self.test_directory = test_directory

    DEFAULT_RESULTS_DIRECTORY: str = "./results"
    """The default results directory."""

    DEFAULT_CLASSIFIERS: List[ClassificationEvaluator] = [
        DecisionTreeClassificationEvaluator()
    ]

    INDIVIDUAL_RESULT_COLUMNS: List[str] = [
        "TC Collection",
        "TC Group",
        "TC Size Group",
        "TC Actual Size",
        "Classifier",
        "Accuracy",
        "# missing URLs",
    ]

    def evaluate(
        self,
        vector_files: List[str],
        result_directory: str = DEFAULT_RESULTS_DIRECTORY,
        classifiers: List[ClassificationEvaluator] = None,
        tc_collection: Set[str] = None,
        tc: Set[str] = None,
        sub_tc: Set[str] = None,
    ) -> None:
        """Main evaluation function. Note that the test directory has already been set.

        Parameters
        ----------
        vector_files : str
            The vector txt file. Must be utf-8 encoded.
        result_directory : str
            Optional. The result directory. The directory must not exist (otherwise, the method will not be executed).
        classifiers : List[ClassificationEvaluator]
            Optional. The classifiers that shall be used.
        tc_collection : Set[str]
            Optional. The test case collection, e.g. "tc4"
        tc : Set[str]
            Optional. The actual test case, e.g. "people".
        sub_tc : Set[str]
            Optional. The sub test case, typically a number indicating the size of the test
            case such as "500".

        Returns
        -------
            None
        """

        # check whether the result directory exists
        result_directory_path = Path(result_directory)
        if result_directory_path.exists():
            logger.error(
                f"""The specified results directory exists already. 
                    Specified directory: {result_directory_path.resolve()} 
                    Aborting program!"""
            )
            return

        # check whether test directory existence
        test_directory_path = Path(self.test_directory)
        if not test_directory_path.exists():
            logger.error(
                f"The test directory does not exist.\n"
                f"Specified directory: {test_directory_path.resolve()}\n"
                f"Aborting program!"
            )
            return

        # check whether test directory is a directory
        if test_directory_path.is_file():
            logger.error(
                "The test directory must be a directory not a file! Aborting program!"
            )
            return

        if classifiers is None:
            logger.info(
                "Using default classifiers (EvaluationManager.DEFAULT_CLASSIFIERS)."
            )
            classifiers = EvaluationManager.DEFAULT_CLASSIFIERS

        result = pd.DataFrame(
            columns=EvaluationManager.INDIVIDUAL_RESULT_COLUMNS, index=None
        )

        # loop over all vector files:
        for vector_file in vector_files:
            vector_map = self.read_vector_txt_file(vector_file=vector_file)

            # loop over all classifiers
            for classifier in classifiers:
                classifier_result = self.evaluate_vector_map(
                    vector_map=vector_map,
                    classifier=classifier,
                    tc_collection=tc_collection,
                    tc=tc,
                    sub_tc=sub_tc,
                )
                result = result.append(other=classifier_result, ignore_index=True)

        result.to_csv(
            path_or_buf=result_directory_path.joinpath("individual_results.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )

    def evaluate_vector_map(
        self,
        vector_map: Dict[str, np.ndarray],
        classifier: ClassificationEvaluator,
        tc_collection: Set[str] = None,
        tc: Set[str] = None,
        sub_tc: Set[str] = None,
    ) -> pd.DataFrame:
        """Evaluate a single vector map.

        Parameters
        ----------
        vector_map : Dict[str, np.ndarray]
            Vectors (more precisely: a set of vectors) to be evaluated.
        classifier : ClassificationEvaluator
            The classifier to be used on the set of vectors.
        tc_collection : Set[str]
            Optional. The test case collections, e.g. ["tc4"]
        tc : Set[str]
            Optional. The actual test cases, e.g. ["people"].
        sub_tc : Set[str]
            Optional. The sub test case, typically a number indicating the size group of the test
            case such as ["500"].

        Returns
        -------
            A dataframe containing the result statistics.
        """

        result = pd.DataFrame(
            columns=EvaluationManager.INDIVIDUAL_RESULT_COLUMNS, index=None
        )

        # jump into every test case
        for tc_collection_dir in Path.iterdir(Path(self.test_directory)):
            # example for tc_collection_dir: "tc1"
            if tc_collection is not None and tc_collection != "":
                if tc_collection_dir.name not in tc_collection:
                    logger.info(
                        f"Skipping '{tc_collection_dir.resolve()}' (not in tc_collection)"
                    )
                    continue
            for tc_dir in Path.iterdir(Path(tc_collection_dir)):
                # example for tc_dir: "person"
                if tc is not None and tc != "":
                    if tc_dir.name not in tc:
                        logger.info(f"Skipping '{tc_dir.resolve()}' (not in tc)")
                        continue
                for sub_tc_dir in Path.iterdir(Path(tc_dir)):
                    # example for sub_tc_dir: "500"
                    if sub_tc is not None and sub_tc != "":
                        if sub_tc_dir.name not in sub_tc:
                            logger.info(
                                f"Skipping '{sub_tc_dir.resolve()}' (not in sub_tc)"
                            )
                            continue
                    train_test_path = sub_tc_dir.joinpath("train_test")
                    if not train_test_path.exists():
                        logger.warning(
                            f"Could not find '{train_test_path}'. Continue evaluation."
                        )
                        continue
                    eval_result = classifier.evaluate(
                        data_directory=train_test_path, vectors=vector_map
                    )
                    result_row = pd.Series(
                        [
                            # TC Collection:
                            tc_collection_dir.name,
                            # TC Group:
                            tc_dir.name,
                            # TC Size Group:
                            sub_tc_dir.name,
                            # TC Actual Size:
                            eval_result.gs_size,
                            # Classifier
                            eval_result.classifier_name,
                            # Accuracy
                            eval_result.accuracy,
                            # # missing URLs
                            eval_result.number_missed,
                        ],
                        index=result.columns,
                    )
                    result = result.append(result_row, ignore_index=True)
        return result

    def write_uris_of_interest_to_file(self, file_to_write: str) -> None:
        """Write relevant URIs to an UTF-8 encoded file (one URI per line).

        Parameters
        ----------
        file_to_write: str
            The file that shall be written.

        Returns
        -------
            None
        """
        self.write_set_to_file(
            set_to_write=self.get_uris_of_interest(), file_to_write=file_to_write
        )

    def get_uris_of_interest(self) -> Set[str]:
        result = set()

        # derive URIs from test_directory
        for tc_collection_dir in Path.iterdir(Path(self.test_directory)):
            # example for tc_collection_dir: "tc1"
            for tc_dir in Path.iterdir(Path(tc_collection_dir)):
                # example for tc_dir: "person"
                for sub_tc_dir in Path.iterdir(Path(tc_dir)):
                    # example for sub_tc_dir: "500"

                    # positives
                    positive_file_path = sub_tc_dir.joinpath("positives.txt").resolve()
                    result.update(self.read_iris_from_file(str(positive_file_path)))

                    # negatives
                    negative_file_path = sub_tc_dir.joinpath("negatives.txt").resolve()
                    result.update(self.read_iris_from_file(str(negative_file_path)))

                    # hard negatives
                    hard_negatives_path = sub_tc_dir.joinpath("negatives_hard.txt")
                    if hard_negatives_path.is_file():
                        result.update(
                            self.read_iris_from_file(str(hard_negatives_path.resolve()))
                        )
        return result

    @classmethod
    def read_iris_from_file(cls, file_to_read_from: str) -> Set[str]:
        """Read the IRIs from a UTF-8 encoded file where one IRI is written per line.

        Parameters
        ----------
        file_to_read_from : str
            String path to the file.

        Returns
        -------
            A set of (str) IRIs.
        """
        result = set()
        with open(file_to_read_from, encoding="utf-8") as uri_file:
            for line in uri_file:
                line = line.replace("\n", "").replace("\r", "")
                line.strip()
                result.add(line)
        return result

    def write_set_to_file(self, set_to_write: Set[str], file_to_write: str) -> None:
        """Write ethe provided set to file (one set item per line).

        Parameters
        ----------
        set_to_write : str
        file_to_write : str

        Returns
        -------
            None
        """
        with open(file_to_write, "w+", encoding="utf-8") as f:
            for element in set_to_write:
                f.write(element + "\n")

    @classmethod
    def read_vector_txt_file(cls, vector_file: str) -> Dict[str, np.ndarray]:
        result = {}
        with open(vector_file, "r", encoding="utf-8") as vector_file:
            for line in vector_file:
                line = line.replace("\n", "").replace("\r", "")
                line = line.strip()
                elements = line.split(" ")
                if len(elements) > 2:
                    result[elements[0]] = np.array(elements[1:]).astype(float)
                else:
                    logger.warning("Empty line or line with only 2 elements!")
        return result

    @classmethod
    def reduce_vectors(
        cls,
        original_vector_file: str,
        reduced_vector_file_to_write: str,
        entities_of_interest: Union[Set[str], str],
    ) -> None:
        """

        Parameters
        ----------
        original_vector_file : str
        reduced_vector_file_to_write : str
        entities_of_interest : Union[Set[str], str]
            Either a file path to the file that contains the node of interest or a set of already parsed nodes of
            interest.

        Returns
        -------
            None
        """
        if type(entities_of_interest) == str:
            entities_of_interest = cls.read_iris_from_file(
                file_to_read_from=entities_of_interest
            )

        with open(
            reduced_vector_file_to_write, "w+", encoding="utf-8"
        ) as file_to_write:

            def process_with_encoding(encoding: str):
                with open(original_vector_file, "r", encoding=encoding) as file_to_read:
                    for line in file_to_read:
                        line_elements = line.split(" ")
                        if len(line_elements) > 2:
                            if line_elements[0] in entities_of_interest:
                                file_to_write.write(line)

            try:
                process_with_encoding(encoding="utf-8")
            except UnicodeDecodeError as ude:
                logger.error("An unicode error occurred. Trying latin-1 next.")
                process_with_encoding(encoding="latin-1")
