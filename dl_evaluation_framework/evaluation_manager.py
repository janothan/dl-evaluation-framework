import logging.config
from pathlib import Path

# some logging configuration
from typing import Dict, List, Set, Union

import numpy as np

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

    DEFAULT_RESULTS_DIRECTORY = "./results"
    """The default results directory."""

    def evaluate(
        self,
        vector_file: str,
        result_directory: str = DEFAULT_RESULTS_DIRECTORY,
        tc_collection: str = "",
        tc: str = "",
        sub_tc="",
    ) -> None:
        """Main evaluation function.

        Parameters
        ----------
        vector_file : str
            The vector txt file. Must be utf-8 encoded.
        result_directory : str
            Optional. The result directory. The directory must not exist (otherwise, the method will not be executed).
        tc_collection : str
            Optional. The test case collection, e.g. "tc4"
        tc : str
            Optional. The actual test case, e.g. "people".
        sub_tc : str
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

        self.vector_map = self.read_vector_txt_file(vector_file=vector_file)

        pass

    def write_uris_of_interest_to_file(self, file_to_write: str) -> None:
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
            with open(original_vector_file, "r", encoding="utf-8") as file_to_read:
                for line in file_to_read:
                    line_elements = line.split(" ")
                    if len(line_elements) > 2:
                        if line_elements[0] in entities_of_interest:
                            file_to_write.write(line + "\n")