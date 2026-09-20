from src.classifiers.base import ArticleClassifier
from src.classifiers.schema import ClassificationResult


class LLMClassifier(ArticleClassifier):

    def classify(self, record) -> ClassificationResult:
        raise NotImplementedError