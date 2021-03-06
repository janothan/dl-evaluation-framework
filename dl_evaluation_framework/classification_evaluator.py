from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union, List, Set
import numpy as np
from pandas.errors import EmptyDataError
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    missed: Set[str]

    @property
    def number_missed(self):
        return len(self.missed)

    classifier_name: str
    data_directory: str
    gs_size: int
    """Size of the gold standard, i.e. |positives| + |negatives|. Must be an even number.
    """


class ClassificationEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        """Perform an evaluation on the specified directory using the specified vectors.

        Parameters
        ----------
        data_directory : str
            The directory where the test and train files reside.
        vectors : Dict[str, np.ndarray]
            The vectors that are to be used for the evaluation.

        Returns
        -------
            EvaluationResult
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
            concept = ClassificationEvaluator.remove_tags(row[0])
            if concept in vectors:
                features.append(vectors[concept])
                labels.append(row[1])
            else:
                missed.add(concept)

        return FeatureLabelTuple(features=features, labels=labels, missed=missed)

    @staticmethod
    def remove_tags(input_str: str) -> str:
        input_str = input_str.lstrip("<")
        input_str = input_str.rstrip(">")
        return input_str

    def evaluate_with_classifier(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
        classifier,
        classifier_name: str,
    ) -> Union[None, EvaluationResult]:
        if type(data_directory) == str:
            data_directory = Path(data_directory)

        train_test: TrainTestTuple = self.parse_train_test(
            data_directory=data_directory
        )

        missed: Set[str] = set()

        if train_test.train is None:
            train_size = 0
        else:
            train_size = len(train_test.train.index)

        if train_test.test is None:
            test_size = 0
        else:
            test_size = len(train_test.test.index)
        gs_size: int = train_size + test_size

        if gs_size == 0:
            logger.error("The size of the gold standard is 0. Return None.")
            return None

        # train
        features_labels_train = self.prepare_for_ml(
            vectors=vectors, label_df=train_test.train
        )

        try:
            classifier.fit(
                X=features_labels_train.features, y=features_labels_train.labels
            )
        except Exception as e:
            logger.error(
                "An error occurred while trying to fit the classifier. Return None.", e
            )
            return None
        missed.update(features_labels_train.missed)

        # test
        features_labels_test = self.prepare_for_ml(
            vectors=vectors, label_df=train_test.test
        )
        accuracy = classifier.score(
            X=features_labels_test.features, y=features_labels_test.labels
        )
        missed.update(features_labels_test.missed)

        return EvaluationResult(
            accuracy=accuracy,
            missed=missed,
            classifier_name=classifier_name,
            data_directory=str(data_directory.resolve()),
            gs_size=gs_size,
        )


class DecisionTreeClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier_name: str = "Decision_Tree"

    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        classifier = DecisionTreeClassifier()
        return super().evaluate_with_classifier(
            data_directory=data_directory,
            vectors=vectors,
            classifier=classifier,
            classifier_name=self.classifier_name,
        )


class NaiveBayesClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier_name: str = "Naive_Bayes"

    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        classifier = GaussianNB()
        return super().evaluate_with_classifier(
            data_directory=data_directory,
            vectors=vectors,
            classifier=classifier,
            classifier_name=self.classifier_name,
        )


class KnnClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier_name: str = "KNN"

    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        classifier = KNeighborsClassifier()
        return super().evaluate_with_classifier(
            data_directory=data_directory,
            vectors=vectors,
            classifier=classifier,
            classifier_name=self.classifier_name,
        )


class SvmClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier_name: str = "SVM"

    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        classifier = LinearSVC()
        return super().evaluate_with_classifier(
            data_directory=data_directory,
            vectors=vectors,
            classifier=classifier,
            classifier_name=self.classifier_name,
        )


class RandomForrestClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier_name: str = "Random_Forrest"

    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        classifier = RandomForestClassifier()
        return super().evaluate_with_classifier(
            data_directory=data_directory,
            vectors=vectors,
            classifier=classifier,
            classifier_name=self.classifier_name,
        )


class MlpClassificationEvaluator(ClassificationEvaluator):
    def __init__(self):
        self.classifier_name: str = "Multi_Layer_Perceptron"

    def evaluate(
        self,
        data_directory: Union[str, Path],
        vectors: Dict[str, np.ndarray],
    ) -> EvaluationResult:
        classifier = MLPClassifier()
        return super().evaluate_with_classifier(
            data_directory=data_directory,
            vectors=vectors,
            classifier=classifier,
            classifier_name=self.classifier_name,
        )
