import logging.config
from dataclasses import dataclass
from pathlib import Path
from dl_evaluation_framework.classification_evaluator import (
    ClassificationEvaluator,
    DecisionTreeClassificationEvaluator,
    NaiveBayesClassificationEvaluator,
    KnnClassificationEvaluator,
    SvmClassificationEvaluator,
    RandomForrestClassificationEvaluator,
    MlpClassificationEvaluator,
)

# some logging configuration
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

logconf_file = Path.joinpath(Path(__file__).parent.resolve(), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorTuple:
    vector_name: str
    vector_path: str


@dataclass(frozen=True)
class ResultMissingTuple:
    result: pd.DataFrame
    """Dataframe has columns: EvaluationManager.INDIVIDUAL_RESULT_COLUMNS
    """

    missing: Set[str]


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
        DecisionTreeClassificationEvaluator(),
        NaiveBayesClassificationEvaluator(),
        KnnClassificationEvaluator(),
        SvmClassificationEvaluator(),
        RandomForrestClassificationEvaluator(),
        MlpClassificationEvaluator(),
    ]

    MISSING_COLUMNS: List[str] = [
        "Missing URL",
        "Vector Name",
        "Vector Path",
    ]

    INDIVIDUAL_RESULT_COLUMNS: List[str] = [
        "TC Collection",
        "TC Group",
        "TC Size Group",
        "TC Actual Size",
        "Vector Name",
        "Classifier",
        "Accuracy",
        "# missing URLs",
        "Vector Path",
    ]

    TCC_RESULT_COLUMNS: List[str] = [
        "TC Collection",
        "Size Group",
        "Vector Name",
        "Classifier",
        "AVG Accuracy",
        "AVG # missing URLs",
        "Vector Path",
    ]

    TCG_RESULT_COLUMNS: List[str] = [
        "TC Group",
        "TC Size Group",
        "AVG TC Actual Size",
        "Vector Name",
        "Classifier",
        "AVG Accuracy",
        "AVG # missing URLs",
        "Vector Path",
    ]

    def evaluate(
        self,
        vector_names_and_files: List[VectorTuple],
        result_directory: str = DEFAULT_RESULTS_DIRECTORY,
        classifiers: List[ClassificationEvaluator] = None,
        tc_collection: Set[str] = None,
        tc: Set[str] = None,
        sub_tc: Set[str] = None,
    ) -> None:
        """Main evaluation function. Note that the test directory has already been set.

        Parameters
        ----------
        vector_names_and_files : VectorTuple
            The vector txt file (must be utf-8 or latin-1 encoded) and vector name.
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

        # check whether test directory exists
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

        individual_result = pd.DataFrame(
            columns=EvaluationManager.INDIVIDUAL_RESULT_COLUMNS, index=None
        )

        missing_urls = pd.DataFrame(
            columns=EvaluationManager.MISSING_COLUMNS, index=None
        )

        # loop over all vector files:
        for vector_tuple in vector_names_and_files:
            vector_map = self.read_vector_txt_file(vector_file=vector_tuple.vector_path)

            missing_set: Set[str] = set()

            # loop over all classifiers
            for classifier in classifiers:
                classifier_result = self.evaluate_vector_map(
                    vector_map=vector_map,
                    vector_tuple=vector_tuple,
                    classifier=classifier,
                    tc_collection=tc_collection,
                    tc=tc,
                    sub_tc=sub_tc,
                )

                if classifier_result is None:
                    logger.warning(
                        f"No classifier result for {vector_tuple.vector_name}!"
                    )
                    continue
                individual_result = pd.concat(
                    objs=[individual_result, classifier_result.result],
                    ignore_index=True,
                )
                missing_set.update(classifier_result.missing)

            # persisting missing urls per vector set:
            for missing_url in missing_set:
                missing_row: pd.DataFrame = pd.DataFrame(
                    data=[
                        (
                            # "Missing URL"
                            missing_url,
                            # "Vector Name"
                            vector_tuple.vector_name,
                            # "Vector Path"
                            vector_tuple.vector_path,
                        )
                    ],
                    columns=missing_urls.columns,
                )
                missing_urls = pd.concat([missing_urls, missing_row], ignore_index=True)

        logger.info(f"Making results directory: {result_directory}")
        result_directory_path.mkdir()

        # write individual results to disk
        individual_result.to_csv(
            path_or_buf=result_directory_path.joinpath("individual_results.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )

        # write missing urls to disk
        missing_urls.to_csv(
            path_or_buf=result_directory_path.joinpath("missing_urls.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )

        # write aggregate/derived results
        EvaluationManager.write_aggregate_files(
            individual_result=individual_result,
            result_directory_path=result_directory_path,
        )

        logger.info(f"Done writing the results. Check folder {result_directory_path}")

    @staticmethod
    def write_aggregate_files(
        individual_result: pd.DataFrame, result_directory_path: Path
    ) -> None:
        """Write derived evaluation files from individual_result.

        Parameters
        ----------
        individual_result : pd.DataFrame
            The dataframe on which the aggregation is based on. The frame has columns defined in
            INDIVIDUAL_RESULT_COLUMNS.
        result_directory_path : Path

        Returns
        -------
            None
        """

        # write test case collection aggregation to disk
        tcc_aggregate_frame = EvaluationManager.calculate_tcc_aggregate_frame(
            individual_result=individual_result
        )
        tcc_aggregate_frame.to_csv(
            path_or_buf=result_directory_path.joinpath("tc_collection_results.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )

        best_tcc_aggregate_frame = EvaluationManager.calculate_best_tcc_aggregate_frame(
            tcc_aggregate_frame=tcc_aggregate_frame
        )
        best_tcc_aggregate_frame.to_csv(
            path_or_buf=result_directory_path.joinpath(
                "best_tc_collection_results.csv"
            ),
            index=False,
            header=True,
            encoding="utf-8",
        )

        best_tcc_comparison_frame = (
            EvaluationManager.best_tc_collection_comparison_results(
                best_tc_collection_results=best_tcc_aggregate_frame,
                is_highlight_best=True,
            )
        )
        best_tcc_comparison_frame.to_csv(
            path_or_buf=result_directory_path.joinpath(
                "comparison_best_tc_collection_results.csv"
            ),
            index=False,
            header=True,
            encoding="utf-8",
        )

        # write the test case group aggregation to disk
        tcg_aggregate_frame = EvaluationManager.calculate_tcg_aggregate_frame(
            individual_result=individual_result
        )
        tcg_aggregate_frame.to_csv(
            path_or_buf=result_directory_path.joinpath("tc_group_results.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )

        best_tcg_aggregate_frame: pd.DataFrame = (
            EvaluationManager.calculate_best_tcg_aggregate_frame(
                tcg_agg_result=tcg_aggregate_frame
            )
        )
        best_tcg_aggregate_frame.to_csv(
            path_or_buf=result_directory_path.joinpath("best_tc_group_results.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )

        best_tcg_comparison_frame = EvaluationManager.best_tc_group_comparison_results(
            best_tc_group_results=best_tcg_aggregate_frame, is_highlight_best=True
        )
        best_tcg_comparison_frame.to_csv(
            path_or_buf=result_directory_path.joinpath(
                "comparison_best_tc_group_results.csv"
            ),
            index=False,
            header=True,
            encoding="utf-8",
        )

    @staticmethod
    def calculate_best_tcg_aggregate_frame(
        tcg_agg_result: pd.DataFrame,
    ) -> pd.DataFrame:
        best_tcg_agg_result = pd.DataFrame(
            columns=EvaluationManager.TCG_RESULT_COLUMNS, index=None
        )
        for tcg in tcg_agg_result["TC Group"].unique():
            tcg_frame = tcg_agg_result.loc[tcg_agg_result["TC Group"] == tcg]

            for size_group in tcg_frame["TC Size Group"].unique():
                tcg_size_frame = tcg_frame.loc[tcg_frame["TC Size Group"] == size_group]

                for vname in tcg_size_frame["Vector Name"].unique():
                    tcg_vname_frame = tcg_size_frame.loc[
                        tcg_size_frame["Vector Name"] == vname
                    ]

                    best_acc: float = 0.0
                    best_row: pd.DataFrame = pd.DataFrame()
                    for classifier in tcg_vname_frame["Classifier"].unique():
                        tcg_classifier_frame = tcg_vname_frame.loc[
                            tcg_vname_frame["Classifier"] == classifier
                        ]

                        currenct_acc = tcg_classifier_frame.iloc[0]["AVG Accuracy"]
                        if currenct_acc > best_acc:
                            best_row: pd.DataFrame = tcg_classifier_frame
                            best_acc = currenct_acc

                    r, _ = best_row.shape
                    if r == 0:
                        logger.warning(
                            f"No best row found for vector '{vname}' on TCC {tcg}."
                        )
                    else:
                        best_tcg_agg_result = pd.concat(
                            objs=[best_tcg_agg_result, best_row], ignore_index=True
                        )
        return best_tcg_agg_result

    @staticmethod
    def best_tc_group_comparison_results(
        best_tc_group_results: pd.DataFrame, is_highlight_best: bool
    ) -> pd.DataFrame:
        result_columns = ["TC Group", "TC Size Group"]

        for vector_name in best_tc_group_results["Vector Name"].unique():
            result_columns.append(vector_name)

        result_df = pd.DataFrame(columns=result_columns)

        for tc_collection in best_tc_group_results["TC Group"].unique():
            tcg_frame = best_tc_group_results.loc[
                best_tc_group_results["TC Group"] == tc_collection
            ]
            for size_group in tcg_frame["TC Size Group"].unique():
                tcc_size_frame = tcg_frame.loc[tcg_frame["TC Size Group"] == size_group]

                result_row_data = [tc_collection, size_group]

                for vector_name in result_columns[2:]:
                    result_row_data.append(
                        tcc_size_frame.loc[
                            tcc_size_frame["Vector Name"] == vector_name
                        ].iloc[0]["AVG Accuracy"]
                    )
                result_row: pd.DataFrame = pd.DataFrame(
                    data=[result_row_data], columns=result_columns, index=None
                )
                result_df = pd.concat(
                    objs=[result_df, result_row],
                    ignore_index=True,
                )
        if is_highlight_best:
            for idx, row in result_df.iterrows():
                best_acc: float = 0.0
                best_vname: str = ""
                for vname in result_columns[2:]:
                    current_acc = result_df.iloc[idx][vname]
                    if current_acc > best_acc:
                        best_acc = current_acc
                        best_vname = vname
                if best_vname == "":
                    print("No best value found!")
                else:
                    result_df.iloc[idx][best_vname] = f"** {best_acc} **"

        return result_df

    @staticmethod
    def calculate_tcg_aggregate_frame(individual_result: pd.DataFrame) -> pd.DataFrame:
        tcg_agg_result = pd.DataFrame(
            columns=EvaluationManager.TCG_RESULT_COLUMNS, index=None
        )

        for tcg in individual_result["TC Group"].unique():
            tcg_frame = individual_result.loc[individual_result["TC Group"] == tcg]

            for size_group in tcg_frame["TC Size Group"].unique():
                tcg_size_frame = tcg_frame.loc[tcg_frame["TC Size Group"] == size_group]

                for classifier in tcg_size_frame["Classifier"].unique():

                    tcg_size_classifier_frame = tcg_size_frame.loc[
                        tcg_size_frame["Classifier"] == classifier
                    ]

                    for vector_name in tcg_size_classifier_frame[
                        "Vector Name"
                    ].unique():
                        tcg_size_classifier_vname_frame = tcg_size_frame.loc[
                            tcg_size_frame["Vector Name"] == vector_name
                        ]

                        mean_tc_act_size = tcg_size_classifier_vname_frame[
                            "TC Actual Size"
                        ].mean()
                        mean_acc = tcg_size_classifier_vname_frame["Accuracy"].mean()
                        mean_missing_urls = tcg_size_classifier_vname_frame[
                            "# missing URLs"
                        ].mean()

                        vector_file_path = tcg_size_classifier_vname_frame.iloc[0][
                            "Vector Path"
                        ]

                        result_row = pd.DataFrame(
                            data=[
                                (
                                    # "TC Group"
                                    tcg,
                                    # "TC Size Group"
                                    size_group,
                                    # "AVG TC Actual Size"
                                    mean_tc_act_size,
                                    # "Vector Name"
                                    vector_name,
                                    # "Classifier"
                                    classifier,
                                    # "Accuracy"
                                    mean_acc,
                                    # "AVG # missing URLs"
                                    mean_missing_urls,
                                    # "Vector Path"
                                    vector_file_path,
                                )
                            ],
                            columns=tcg_agg_result.columns,
                        )

                        tcg_agg_result = pd.concat(
                            [result_row, tcg_agg_result], ignore_index=True
                        )
        return tcg_agg_result

    @staticmethod
    def calculate_best_tcc_aggregate_frame(
        tcc_aggregate_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate according to the best ML algorithm given the already TCC aggregated data frame.

        Parameters
        ----------
        tcc_aggregate_frame: pd.DataFrame
            Should have columns EvaluationManager.TCC_RESULT_COLUMNS.

        Returns
        -------
            Aggregated dataframe with columns EvaluationManager.TCC_RESULT_COLUMNS.
        """
        best_tcc_agg_result = pd.DataFrame(
            columns=EvaluationManager.TCC_RESULT_COLUMNS, index=None
        )

        unique_tcc = tcc_aggregate_frame["TC Collection"].unique()
        for tc_collection in unique_tcc:
            logger.debug(f"[best tcc] Working on {tc_collection}...")
            tc_collection_frame = tcc_aggregate_frame.loc[
                tcc_aggregate_frame["TC Collection"] == tc_collection
            ]

            for size_group in tc_collection_frame["Size Group"].unique():
                logger.debug(f"[best tcc] ... for size {size_group}")
                tcc_size_frame = tc_collection_frame.loc[
                    tc_collection_frame["Size Group"] == size_group
                ]
                for vname in tcc_size_frame["Vector Name"].unique():
                    logger.debug(f"[best tcc] ... for vector name {vname}")
                    tcc_vname = tcc_size_frame.loc[
                        tcc_size_frame["Vector Name"] == vname
                    ]

                    best_row: pd.DataFrame = pd.DataFrame()
                    best_acc: float = 0

                    for vclassifier in tcc_vname["Classifier"].unique():
                        tcc_vclassifier = tcc_vname.loc[
                            tcc_vname["Classifier"] == vclassifier
                        ]
                        currenct_acc = tcc_vclassifier.iloc[0]["AVG Accuracy"]
                        if currenct_acc > best_acc:
                            best_row: pd.DataFrame = tcc_vclassifier
                            best_acc = currenct_acc

                    r, _ = best_row.shape
                    if r == 0:
                        logger.warning(
                            f"No best row found for vector '{vname}' on TCC {tc_collection}."
                        )
                    else:
                        best_tcc_agg_result = pd.concat(
                            objs=[best_tcc_agg_result, best_row], ignore_index=True
                        )
        return best_tcc_agg_result

    @staticmethod
    def best_tc_collection_comparison_results(
        best_tc_collection_results: pd.DataFrame, is_highlight_best: bool
    ) -> pd.DataFrame:
        """Calculates a dataframe from the best tcc results so that the best accuracies for each vector name
        can be directly compared.

        Parameters
        ----------
        best_tc_collection_results: pd.DataFrame
            DataFrame with columns EvaluationManager.TCC_RESULT_COLUMNS obtained by calling
            calculate_best_tcc_aggregate_frame(...)!
        is_highlight_best: bool
            If true the best accuracy in each row is highlighted using stars.

        Returns
        -------
            A dataframe that allows to compare the best performance per test case collection.
        """
        result_columns = ["TC Collection", "Size Group"]

        for vector_name in best_tc_collection_results["Vector Name"].unique():
            result_columns.append(vector_name)

        result_df = pd.DataFrame(columns=result_columns)

        for tc_collection in best_tc_collection_results["TC Collection"].unique():
            tcc_frame = best_tc_collection_results.loc[
                best_tc_collection_results["TC Collection"] == tc_collection
            ]
            for size_group in tcc_frame["Size Group"].unique():
                tcc_size_frame = tcc_frame.loc[tcc_frame["Size Group"] == size_group]

                result_row_data = [tc_collection, size_group]

                for vector_name in result_columns[2:]:
                    result_row_data.append(
                        tcc_size_frame.loc[
                            tcc_size_frame["Vector Name"] == vector_name
                        ].iloc[0]["AVG Accuracy"]
                    )
                result_row: pd.DataFrame = pd.DataFrame(
                    data=[result_row_data], columns=result_columns, index=None
                )
                result_df = pd.concat(
                    objs=[result_df, result_row],
                    ignore_index=True,
                )
        if is_highlight_best:
            for idx, row in result_df.iterrows():
                best_acc: float = 0.0
                best_vname: str = ""
                for vname in result_columns[2:]:
                    current_acc = result_df.iloc[idx][vname]
                    if current_acc > best_acc:
                        best_acc = current_acc
                        best_vname = vname
                if best_vname == "":
                    print("No best value found!")
                else:
                    result_df.iloc[idx][best_vname] = f"** {best_acc} **"

        return result_df

    @staticmethod
    def calculate_tcc_aggregate_frame(individual_result: pd.DataFrame) -> pd.DataFrame:
        """Given the individual_result data frame, this method creates an aggregated frame based on
        test case collection, group size, and classifier.

        Parameters
        ----------
        individual_result : pd.DataFrame
            The dataframe on which the aggregation is based on.

        Returns
        -------
            Aggregated dataframe with columns EvaluationManager.TCC_RESULT_COLUMNS.
        """
        tcc_agg_result = pd.DataFrame(
            columns=EvaluationManager.TCC_RESULT_COLUMNS, index=None
        )
        unique_tcc = individual_result["TC Collection"].unique()
        for tc_collection in unique_tcc:
            logger.debug(f"Working on {tc_collection}...")
            tc_collection_frame = individual_result.loc[
                individual_result["TC Collection"] == tc_collection
            ]

            for size_group in tc_collection_frame["TC Size Group"].unique():
                logger.debug(f"... for size {size_group}")
                tcc_size_frame = tc_collection_frame.loc[
                    tc_collection_frame["TC Size Group"] == size_group
                ]

                for classifier in tcc_size_frame["Classifier"].unique():
                    logger.debug(f"... for classifier {classifier}")
                    tcc_size_classifier_frame = tcc_size_frame.loc[
                        tcc_size_frame["Classifier"] == classifier
                    ]

                    for vector_name in tcc_size_classifier_frame[
                        "Vector Name"
                    ].unique():
                        logger.debug(f"... for vector name {vector_name}")
                        tcc_size_classifier_vname_frame = tcc_size_classifier_frame[
                            tcc_size_classifier_frame["Vector Name"] == vector_name
                        ]

                        acc_mean = tcc_size_classifier_vname_frame["Accuracy"].mean()
                        missing_url_mean = tcc_size_classifier_vname_frame[
                            "# missing URLs"
                        ].mean()

                        vector_file_path = tcc_size_classifier_vname_frame.iloc[0][
                            "Vector Path"
                        ]

                        result_row = pd.DataFrame(
                            [
                                (
                                    # "TC Collection"
                                    tc_collection,
                                    # "Size Group"
                                    size_group,
                                    # "Vector Name"
                                    vector_name,
                                    # "Classifier"
                                    classifier,
                                    # "AVG Accuracy"
                                    acc_mean,
                                    # "AVG # missing URLs"
                                    missing_url_mean,
                                    # "Vector Path"
                                    vector_file_path,
                                )
                            ],
                            columns=tcc_agg_result.columns,
                        )
                        tcc_agg_result = pd.concat(
                            [tcc_agg_result, result_row], ignore_index=True
                        )
        r, _ = tcc_agg_result.shape
        logger.debug(f"Number of rows of tcc_aggregate_frame {r}")
        return tcc_agg_result

    def evaluate_vector_map(
        self,
        vector_map: Dict[str, np.ndarray],
        vector_tuple: VectorTuple,
        classifier: ClassificationEvaluator,
        tc_collection: Set[str] = None,
        tc: Set[str] = None,
        sub_tc: Set[str] = None,
    ) -> Union[ResultMissingTuple, None]:
        """Evaluate a single vector map.

        Parameters
        ----------
        vector_map : Dict[str, np.ndarray]
            Vectors (more precisely: a set of vectors) to be evaluated.
        vector_tuple : VectorTuple
            Meta information about the vector map.
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
            ResultMissingTuple containing:
            - A dataframe containing the result statistics.
            - A dataframe containing the missing URLs.
        """

        logger.info(f"Evaluating {vector_tuple.vector_name}")

        result: pd.DataFrame = pd.DataFrame(
            columns=EvaluationManager.INDIVIDUAL_RESULT_COLUMNS, index=None
        )

        missing_set: Set[str] = set()

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

                    # train_test:
                    train_test_path = sub_tc_dir.joinpath("train_test")
                    result = EvaluationManager.__add_classifier_result_row(
                        train_test_path=train_test_path,
                        vector_map=vector_map,
                        vector_tuple=vector_tuple,
                        classifier=classifier,
                        missing_set=missing_set,
                        result=result,
                        tc_collection_name=tc_collection_dir.name,
                        tc_dir_name=tc_dir.name,
                        sub_tc_dir_name=sub_tc_dir.name,
                    )

                    train_test_path_hard = sub_tc_dir.joinpath("train_test_hard")
                    if train_test_path_hard.exists():
                        # we do not want warnings if there is no hard case since this may appear often
                        result = EvaluationManager.__add_classifier_result_row(
                            train_test_path=train_test_path_hard,
                            vector_map=vector_map,
                            vector_tuple=vector_tuple,
                            classifier=classifier,
                            missing_set=missing_set,
                            result=result,
                            tc_collection_name=tc_collection_dir.name + "_hard",
                            tc_dir_name=tc_dir.name,
                            sub_tc_dir_name=sub_tc_dir.name,
                        )

        return ResultMissingTuple(result=result, missing=missing_set)

    @staticmethod
    def __add_classifier_result_row(
        train_test_path: Path,
        vector_map: Dict[str, np.ndarray],
        vector_tuple: VectorTuple,
        classifier: ClassificationEvaluator,
        missing_set: Set[str],
        result: pd.DataFrame,
        tc_collection_name: str,
        tc_dir_name: str,
        sub_tc_dir_name: str,
    ) -> Union[pd.DataFrame]:
        if not train_test_path.exists():
            logger.warning(f"Could not find '{train_test_path}'. Continue evaluation.")
            return result

        try:
            eval_result = classifier.evaluate(
                data_directory=train_test_path, vectors=vector_map
            )
        except ValueError as error:
            logger.error(
                f"An error occurred with classifier {classifier} using {vector_tuple.vector_name} "
                f"on directory {train_test_path}. Evaluation continues.",
                error,
            )
            return result

        if eval_result is None:
            logger.warning(
                f"Could not determine result for {vector_tuple.vector_name} "
                f"on {train_test_path}"
            )
            return result

        missing_set.update(eval_result.missed)

        result_row = pd.DataFrame(
            data=[
                (
                    # TC Collection:
                    tc_collection_name,
                    # TC Group:
                    tc_dir_name,
                    # TC Size Group:
                    sub_tc_dir_name,
                    # TC Actual Size:
                    eval_result.gs_size,
                    # "Vector Name"
                    vector_tuple.vector_name,
                    # Classifier
                    eval_result.classifier_name,
                    # Accuracy
                    eval_result.accuracy,
                    # # missing URLs
                    eval_result.number_missed,
                    # "Vector Path"
                    vector_tuple.vector_path,
                )
            ],
            columns=result.columns,
        )
        return pd.concat([result, result_row], axis=0, ignore_index=True)

    def write_uris_of_interest_to_file(self, file_to_write: str) -> None:
        """Write relevant URIs to a UTF-8 encoded file (one URI per line).

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

    @staticmethod
    def write_set_to_file(set_to_write: Set[str], file_to_write: str) -> None:
        """Write ethe provided set to a utf-8 encoded file (one set item per line).

        Parameters
        ----------
        set_to_write : str
            The set that shall be persisted.
        file_to_write : str
            The file that shall be written.

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
                    result[cls.remove_tags(elements[0])] = np.array(
                        elements[1:]
                    ).astype(np.float32)
                else:
                    logger.warning("Empty line or line with only 2 elements!")
        return result

    @staticmethod
    def remove_tags(input_str: str) -> str:
        input_str = input_str.lstrip("<")
        input_str = input_str.rstrip(">")
        return input_str

    @classmethod
    def reduce_vectors(
        cls,
        original_vector_file: str,
        reduced_vector_file_to_write: str,
        entities_of_interest: Union[Set[str], str],
    ) -> None:
        """Write a new vector file that contains only a desired subset of vectors.

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
        if original_vector_file is None or original_vector_file.strip() == "":
            logger.error(
                "Cannot reduce vector file because the provided file path is invalid."
            )
            return

        if entities_of_interest is None or len(entities_of_interest) == 0:
            logger.error(
                "Cannot reduce vector file because the provided entities_of_interest file path is invalid "
                "or the set is empty."
            )
            return

        if type(entities_of_interest) == str:
            entities_of_interest = cls.read_iris_from_file(
                file_to_read_from=entities_of_interest
            )

        def process_with_encoding(encoding: str):
            with open(
                reduced_vector_file_to_write,
                "w+",
                encoding="utf-8",  # note: w+ will overwrite (this is what we want)
            ) as file_to_write:
                with open(original_vector_file, "r", encoding=encoding) as file_to_read:
                    for line in file_to_read:
                        line_elements = line.split(" ")
                        if len(line_elements) > 2:
                            if line_elements[0] in entities_of_interest:
                                file_to_write.write(line)

        try:
            process_with_encoding(encoding="utf-8")
        except UnicodeDecodeError as ude:
            logger.error("A unicode error occurred. Trying latin-1 next.", ude)
            process_with_encoding(encoding="latin-1")
