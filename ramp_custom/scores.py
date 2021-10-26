from rampwf.score_types.base import BaseScoreType


class AveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="average precision", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        return 0
