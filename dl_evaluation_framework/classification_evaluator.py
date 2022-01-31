from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union, List, Set
import numpy as np
from pandas.errors import EmptyDataError
from sklearn import tree
import logging.config
import pandas as pd
from dataclasses import dataclass

logconf_file = Path.joinpath(Path(__file__).parent.resolve(), "log.conf")
logging.config.fileConfig(fname=logconf_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainTestTuple:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureLabelTuple:
    features: List[np.ndarray]
    labels: List[int]
    missed: Set[str]


class ClassificationEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, data_directory: str, vectors: Dict[str, np.ndarray], results_file: str
    ) -> None:
        """

        Parameters
        ----------
        data_directory : str
            The directory where the test and train files reside.
        vectors : Dict[str, np.ndarray]
            The vectors that are to be used for the evaluation.
        results_file : str
            The results file that is to be written.

        Returns
        -------
            None
        """
        pass

    @staticmethod
    def train_test_file_to_df(file: Union[str, Path]) -> Union[pd.DataFrame, None]:
        try:
            return pd.read_csv(file, delimiter="\\s+", header=None)
        except EmptyDataError as e:
            logger.error(f"The provided file '{file}' is empty. Returning None.")
            return None


    @staticmethod
    def parse_train_test(
        data_directory: Union[str, Path]
    ) -> Union[TrainTestTuple, None]:
        data_directory_path = data_directory
        if type(data_directory) == str:
            data_directory_path = Path(data_directory)

        if not data_directory_path.exists():
            logger.error(
                f"The provided data_directory ('{data_directory}') does not exist."
            )
            return None

        train_path = data_directory_path.joinpath("train.txt")
        if not train_path.exists():
            logger.error(
                f"There is not train file. Expected '{train_path.resolve(strict=False)}'"
            )
            return None

        test_path = data_directory_path.joinpath("test.txt")
        if not test_path.exists():
            logger.error(
                f"There is no test file. Expected '{test_path.resolve(strict=False)}'"
            )
            return None

        train_df = ClassificationEvaluator.train_test_file_to_df(train_path)
        test_df = ClassificationEvaluator.train_test_file_to_df(test_path)

        return TrainTestTuple(train=train_df, test=test_df)

    @staticmethod
    def prepare_for_ml(
        vectors: Dict[str, np.ndarray], label_df: pd.DataFrame
    ) -> FeatureLabelTuple:
        features: List[np.ndarray] = []
        labels: List[int] = []
        missed: Set[str] = set()

        for idx, row in label_df.iterrows():
            concept = row[0]
            if concept in vectors:
                features.append(vectors[concept])
                labels.append(concept)
            else:
                missed.add(concept)

        return FeatureLabelTuple(features=features, labels=labels, missed=missed)


class DecisionTreeClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier = tree.DecisionTreeClassifier()

    def evaluate(
        self, data_directory: str, vectors: Dict[str, np.ndarray], results_file: str
    ) -> None:
        train_test = super().parse_train_test(data_directory=data_directory)

        # train
        features_labels = super().prepare_for_ml(vectors=vectors , label_df=train_test.train)

        self.classifier.fit(X=features_labels.features, y=features_labels.labels)


