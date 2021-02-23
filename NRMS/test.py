"""
Online testing module for NRMS
"""

import random
import torch
import pytorch_lightning as pl
import bcolz
import pickle
from glove import generate_glove_vocab
from utils.utils import tokenize_word
import pandas as pd

from models.network import NRMS
from models.deep_cross import DeepCross


class OnlineNRMSModel(pl.LightningModule):
    def __init__(self, hyperParams, file_path):
        super(OnlineNRMSModel, self).__init__()
        self.hyperParams = hyperParams
        self.file_path = file_path
        self.embedding_model = self.load_embedding()
        self.model = NRMS(self.hyperParams["model"], self.embedding_model)
        self.glove_vocab_index = self.load_vocab_glove_index()

    def forward(self, clicks, candidates, topk):
        scores = self.model(clicks, candidates)
        prob, index = scores.topk(topk)
        return index, prob

    def load_embedding(self):
        """
        Load pre-trained glove embedding model
        :return: pre-trained embedding model
        """
        embedding_size = self.hyperParams["model"]["embedding_size"]
        max_vocab_size = self.hyperParams["max_vocab_size"]
        glove_path = self.hyperParams["glove_path"]
        generate_glove_vocab(glove_path, embedding_size, max_vocab_size)
        embeddings = torch.Tensor(bcolz.open(f'{glove_path}/6B.'+str(embedding_size)+'.dat')[:])
        return embeddings

    def load_vocab_glove_index(self):
        """
        load glove vocab dictionary: word -> index
        :return:
        """
        glove_path = self.hyperParams["glove_path"]+"/6B."+str(self.hyperParams["model"]["embedding_size"])+"_idx.pkl"
        return pickle.load(open(glove_path, 'rb'))

    def init_user_behavior(self):
        """

        Returns:
            list: List of user session with userId, clicks and pos/neg impressions
            dict: Dictionary with userId(key) and click history(value)
        """
        path = self.file_path + "/behaviors.tsv"
        user_behavior = pd.read_csv(path, header=None, sep='\t')
        user_behavior.columns = ["Impress_ID",
                                 "User_ID",
                                 "Time",
                                 "History",
                                 "Impressions"]
        userId_clickHis = {}
        session = []
        for i in range(user_behavior.shape[0]):
            userId, clicks, impressions = user_behavior.loc[i, "User_ID"], user_behavior.loc[i, "History"], \
                                          user_behavior.loc[i, "Impressions"]
            if not isinstance(clicks, str):
                continue
            clicks = clicks.split(" ")
            impressions = impressions.split(" ")
            # pos: impression==1, neg: otherwise
            pos, neg = [], []
            for impression in impressions:
                impression_list = impression.split("-")
                if impression_list[1] == "1":
                    pos.append(impression_list[0])
                else:
                    neg.append(impression_list[0])
            userId_clickHis[userId] = clicks
            session.append([userId, clicks, pos, neg])

        return session, userId_clickHis

    def get_news(self):
        path = self.file_path + "/news.tsv"
        news = pd.read_csv(path, header=None, sep='\t')
        news.columns = ["ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_entities",
                        "Abstract_entities"]
        title_token, abstract_token = {}, {}
        for i in range(news.shape[0]):
            id, title, abstract = news.loc[i, "ID"], news.loc[i, "Title"], news.loc[i, "Abstract"]
            if (not isinstance(title, str)) or (not isinstance(abstract, str)):
                continue
            title, abstract = title.lower(), abstract.lower()
            title_token[id] = tokenize_word(title)
            abstract_token[id] = tokenize_word(abstract)

        self.news_df = news
        return title_token, abstract_token

    def get_title_index_dict(self):
        """
       set up a ID-title dictionary
       :return: ID-title dictionary
       """
        title_dict = {}
        for id in self.title_token.keys():
            title_idx = [self._word2idx(text) for text in self.title_token[id]]
            maxLen = self.hyperParams["data"]["wordLen"]
            if len(title_idx) >= maxLen:
                title_idx = title_idx[:maxLen]
            else:
                padding = [0 for _ in range(maxLen-len(title_idx))]
                title_idx = title_idx + padding
            title_dict[id] = title_idx
        return title_dict

    def get_abstract_index_dict(self):
        """
        set up a ID-abstract dictionary
        :return: ID-abstract dictionary
        """
        abstract_dict = {}
        for id in self.abstract_token.keys():
            title_idx = [self._word2idx(text) for text in self.abstract_token[id]]
            maxLen = self.hyperParams["data"]["wordLen"]
            if len(title_idx) >= maxLen:
                title_idx = title_idx[:maxLen]
            else:
                padding = [0 for _ in range(maxLen - len(title_idx))]
                title_idx = title_idx + padding
            abstract_dict[id] = title_idx
        return abstract_dict

    def _word2idx(self, text):
        try:
            index = self.glove_vocab_index[text]
        except:
            index = 0
        return index

    def load_test_data(self):
        self.user_behavior, self.clickHis = self.init_user_behavior()
        self.title_token, self.abstract_token = self.get_news()
        self.title_index_dict, self.abstract_index_dict = self.get_title_index_dict(), self.get_abstract_index_dict()

    def doPrediction(self, news_history, candidates, candidate_num, candidates_index):
        input_candidates = candidates
        news_history, candidates = torch.Tensor(news_history).unsqueeze(0), torch.Tensor(candidates).unsqueeze(0)
        output, score = self(news_history.long(), candidates.long(), candidate_num)
        score = score.squeeze().detach().cpu().tolist()
        prediction = [input_candidates[idx] for idx in output.squeeze()]
        candidates_index_ranking = [candidates_index[idx] for idx in output.squeeze()]

        return prediction, score, candidates_index_ranking


class OnlineDeepCrossNRMSModel(pl.LightningModule):
    def __init__(self, hyperParams, file_path, device='cuda:0'):
        super(OnlineDeepCrossNRMSModel, self).__init__()
        self.hyperParams = hyperParams
        self.file_path = file_path
        self.embedding_model = self.load_embedding()
        self.model = DeepCross(self.hyperParams)
        self.glove_vocab_index = self.load_vocab_glove_index()
        self.to(device)
        self.tensor_device = device

    def forward(self, clicks_title, clicks_abstract, candidate_title, candidate_abstract, topk):
        scores = self.model(clicks_title, clicks_abstract, candidate_title, candidate_abstract)
        prob, index = scores.topk(topk)
        return index, prob

    def load_embedding(self):
        """
        Load pre-trained glove embedding model
        :return: pre-trained embedding model
        """
        embedding_size = self.hyperParams["model"]["embedding_size"]
        max_vocab_size = self.hyperParams["max_vocab_size"]
        glove_path = self.hyperParams["glove_path"]
        generate_glove_vocab(glove_path, embedding_size, max_vocab_size)
        embeddings = torch.Tensor(bcolz.open(f'{glove_path}/6B.'+str(embedding_size)+'.dat')[:])
        return embeddings

    def load_vocab_glove_index(self):
        """
        load glove vocab dictionary: word -> index
        :return:
        """
        glove_path = self.hyperParams["glove_path"]+"/6B."+str(self.hyperParams["model"]["embedding_size"])+"_idx.pkl"
        return pickle.load(open(glove_path, 'rb'))

    def init_user_behavior(self):
        """

        Returns:
            list: List of user session with userId, clicks and pos/neg impressions
            dict: Dictionary with userId(key) and click history(value)
        """
        path = self.file_path + "/behaviors.tsv"
        user_behavior = pd.read_csv(path, header=None, sep='\t')
        user_behavior.columns = ["Impress_ID",
                                 "User_ID",
                                 "Time",
                                 "History",
                                 "Impressions"]
        userId_clickHis = {}
        session = []
        for i in range(user_behavior.shape[0]):
            userId, clicks, impressions = user_behavior.loc[i, "User_ID"], user_behavior.loc[i, "History"], \
                                          user_behavior.loc[i, "Impressions"]
            if not isinstance(clicks, str):
                continue
            clicks = clicks.split(" ")
            impressions = impressions.split(" ")
            # pos: impression==1, neg: otherwise
            pos, neg = [], []
            for impression in impressions:
                impression_list = impression.split("-")
                if impression_list[1] == "1":
                    pos.append(impression_list[0])
                else:
                    neg.append(impression_list[0])
            userId_clickHis[userId] = clicks
            session.append([userId, clicks, pos, neg])

        return session, userId_clickHis

    def get_news(self):
        path = self.file_path + "/news.tsv"
        news = pd.read_csv(path, header=None, sep='\t')
        news.columns = ["ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_entities",
                        "Abstract_entities"]
        title_token, abstract_token = {}, {}
        for i in range(news.shape[0]):
            id, title, abstract = news.loc[i, "ID"], news.loc[i, "Title"], news.loc[i, "Abstract"]
            if (not isinstance(title, str)) or (not isinstance(abstract, str)):
                continue
            title, abstract = title.lower(), abstract.lower()
            title_token[id] = tokenize_word(title)
            abstract_token[id] = tokenize_word(abstract)

        self.news_df = news
        return title_token, abstract_token

    def get_title_index_dict(self):
        """
       set up a ID-title dictionary
       :return: ID-title dictionary
       """
        title_dict = {}
        for id in self.title_token.keys():
            title_idx = [self._word2idx(text) for text in self.title_token[id]]
            maxLen = self.hyperParams["data"]["titleLen"]
            if len(title_idx) >= maxLen:
                title_idx = title_idx[:maxLen]
            else:
                padding = [0 for _ in range(maxLen-len(title_idx))]
                title_idx = title_idx + padding
            title_dict[id] = title_idx
        return title_dict

    def get_abstract_index_dict(self):
        """
        set up a ID-abstract dictionary
        :return: ID-abstract dictionary
        """
        abstract_dict = {}
        for id in self.abstract_token.keys():
            title_idx = [self._word2idx(text) for text in self.abstract_token[id]]
            maxLen = self.hyperParams["data"]["wordLen"]
            if len(title_idx) >= maxLen:
                title_idx = title_idx[:maxLen]
            else:
                padding = [0 for _ in range(maxLen - len(title_idx))]
                title_idx = title_idx + padding
            abstract_dict[id] = title_idx
        return abstract_dict

    def _word2idx(self, text):
        try:
            index = self.glove_vocab_index[text]
        except:
            index = 0
        return index

    def load_test_data(self):
        self.user_behavior, self.clickHis = self.init_user_behavior()
        self.title_token, self.abstract_token = self.get_news()
        self.title_index_dict, self.abstract_index_dict = self.get_title_index_dict(), self.get_abstract_index_dict()

    def doPrediction(self, title_history, abstract_history, candidate_title, candidate_abstract, candidate_num, candidates_index):
        input_candidates = candidate_title
        title_history, candidate_title = torch.Tensor(title_history).unsqueeze(0), torch.Tensor(candidate_title).unsqueeze(0)
        abstract_history, candidate_abstract = torch.Tensor(abstract_history).unsqueeze(0), torch.Tensor(
            candidate_abstract).unsqueeze(0)

        # move to the corresponding device
        title_history, abstract_history, candidate_title, candidate_abstract = \
            title_history.to(self.tensor_device), abstract_history.to(self.tensor_device), \
            candidate_title.to(self.tensor_device), candidate_abstract.to(self.tensor_device)

        output, score = self(title_history.long(), abstract_history.long(), candidate_title.long(), candidate_abstract.long(), candidate_num)
        score = score.squeeze().detach().cpu().tolist()
        prediction = [input_candidates[idx] for idx in output.squeeze()]
        candidates_index_ranking = [candidates_index[idx] for idx in output.squeeze()]

        return prediction, score, candidates_index_ranking


class MINDTest:
    """
    Online testing for MIND dataset given a trained model.
    :param self.model: online testing model
    :param self.user_behavior: history sessions contaning user's [userId, clicks, pos, neg], clicks is a list containing a series of news's Id
    :param self.clickHis: a dictionary whose key is userId and value is user's whole click history
    :param self.title_token: a dictionary containing tokenized new's title, key is new's Id
    :param self.abstract_token: a dictionary containing tokenized new's abstract, key is userId
    :param self.title_index_dict: a dictionary whose key is title token and value is its index in embedding vocabulary
    :param self.abstract_index_dict: a dictionary whose key is abstract token and value is its index in embedding vocabulary
    """
    def __init__(self, hyperParmas, model):
        self.hyperParams = hyperParmas
        self.model = model

    def online_test_title(self, index):
        """
        online inference using new's title
        :param index: user index
        :return:
        """
        # extract user's view history
        news_history_index = self.model.user_behavior[index][1][:50]
        news_history_title_token, news_history_title = [], []
        for idx in news_history_index:
            try:
                news_history_title_token.append(self.model.title_token[idx])
                news_history_title.append(self.model.title_index_dict[idx])
            except:
                continue

        # random select candidate news
        candidate_news_title_index = random.sample(self.model.title_index_dict.keys(), 200)
        candidate_news_title = [self.model.title_index_dict[idx] for idx in candidate_news_title_index]

        # execute the prediction
        result, val, news_ranking_index = self.model.doPrediction(news_history_title, candidate_news_title,
                                                                  len(candidate_news_title_index),
                                                                  candidate_news_title_index)

        # extract the predicted new's title for recommendation
        news_ranking = [self.model.title_token[idx] for idx in news_ranking_index]

        return result, val, news_ranking, news_history_title_token

    def online_test(self, index):
        """
        online inference using new's both title and abstract
        :param index:  user index
        :return:
        """
        # extract user's view history
        news_history_index = self.model.user_behavior[index][1][:50]
        news_history_title_token, news_history_title = [], []
        news_history_abstract_token, news_history_abstract = [], []
        news_category = []
        for idx in news_history_index:
            try:
                news_history_title_token.append(self.model.title_token[idx])
                news_history_title.append(self.model.title_index_dict[idx])
                news_history_abstract_token.append(self.model.abstract_token[idx])
                news_history_abstract.append(self.model.abstract_index_dict[idx])
                news_category.append(self.model.news_df[self.model.news_df["ID"] == idx]["Category"].values[0])
            except:
                continue

        # random select candidate news
        candidate_news_index = random.sample(self.model.abstract_index_dict.keys(), 200)
        candidate_news_title = [self.model.title_index_dict[idx] for idx in candidate_news_index]
        candidate_news_abstract = [self.model.abstract_index_dict[idx] for idx in candidate_news_index]

        # execute the prediction
        result, val, news_ranking_index = self.model.doPrediction(news_history_title, news_history_abstract,
                                                                  candidate_news_title, candidate_news_abstract,
                                                                  len(candidate_news_index), candidate_news_index)

        # extract the predicted new's title for recommendation
        news_ranking = [self.model.title_token[idx] for idx in news_ranking_index]

        rank_category = [self.model.news_df[self.model.news_df["ID"] == idx]["Category"].values[0] for idx in news_ranking_index]

        return result, val, news_ranking, news_history_title_token, rank_category, news_category
