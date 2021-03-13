from nltk.tokenize import RegexpTokenizer
import numpy as np
from collections import defaultdict


def tokenize_word(text):
    if not isinstance(text, str):
        return []
    tokenizer = RegexpTokenizer("[\w]+|[.,!?;|]")
    return tokenizer.tokenize(text)


def dcg_score(y_true, y_score, k=10):
    """
    Discounted Cumulative Gain
    :param y_true: true rank
    :param y_score: actual rank
    :param k: only consider first k elements
    :return: DCG score
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """
    Normalize DCG
    :param y_true: true rank
    :param y_score: actual rank
    :param k: only consider first k elements
    :return: NDCG score
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    """
    Mean reciprocal rank: the reciprocal of the first correct rank
    :param y_true: true rank
    :param y_score: actual rank
    :return: MRR score
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def topk_score(y_true, y_predict, k=3):
    """

    :param y_true: raw, category label
    :param y_predict: predicted categories
    :param k: only top k predictions are accepted
    :return:
    """
    label_map = defaultdict(int)
    for label in y_true:
        label_map[label] += 1
    sorted_category = sorted(label_map.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_category) > k:
        sorted_category = sorted_category[:k]
    label_category = [category for (category, _) in sorted_category]
    dcg = 0.0
    for i, prediction in enumerate(y_predict):
        if label_category.count(prediction) > 0:
            rank = label_category.index(prediction)
            score = 2 ** (len(label_category) - rank - 1)
            if i == 0:
                dcg += score
            else:
                dcg += score / np.log2(i+1)
    norm = 2 ** (k-1)
    for i in range(len(y_predict)-1):
        norm += norm / np.log2(i+2)
    return dcg / norm
