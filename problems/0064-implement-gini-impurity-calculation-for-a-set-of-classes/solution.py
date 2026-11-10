
import numpy as np
from collections import Counter

def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    counts = Counter(y)
    total_count = len(y)
    gini_impurity = 1- np.sum((np.array(list(counts.values()))/total_count) ** 2)

    return gini_impurity
