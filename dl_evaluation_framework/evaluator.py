import logging
import logging.config
from pathlib import Path, PurePath

# some logging configuration
logconf_file = Path.joinpath(Path(__file__).parent.resolve(), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, query_directory: str):
        """Constructor

        Parameters
        ----------
        query_directory : str
            The path to the query directory.
        """
        self.query_directory = query_directory

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

        # check whether the directory exists
        result_directory_path = Path(result_directory)
        if result_directory_path.exists():
            logger.error(
                f"""The specified results directory exists already. 
                    Specified directory: {result_directory_path.resolve()} 
                    Aborting program!"""
            )
            return

        # check query directory existence
        query_directory_path = Path(self.query_directory)
        if not query_directory_path.exists():
            logger.error(
                f"The query directory does not exist.\n"
                f"Specified directory: {query_directory_path.resolve()}\n"
                f"Aborting program!"
            )
            return

        # check whether query directory is a directory
        if query_directory_path.is_file():
            logger.error("The query directory must be a directory not a file! Aborting program!")
            return



        pass

    @classmethod
    def reduce_vectors(
        cls, original_vector_file: str, reduced_vector_file_to_write: str
    ) -> None:
        pass
