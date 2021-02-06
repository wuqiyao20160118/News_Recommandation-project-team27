from nltk.tokenize import RegexpTokenizer
import numpy as np


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
