import pandas as pd
from utils.utils import tokenize_word
import random
import pickle
import torch
from torch.utils import data


class NewsDatasetDCN(data.Dataset):
    def __init__(self, hyperParams, file_path):
        super(NewsDatasetDCN, self).__init__()
        self.hyperParams = hyperParams
        self.file_path = file_path
        self.behavior, self.clickHis = self.init_user_behavior()
        self.title_token, self.abstract_token = self.get_news()
        self.glove_vocab_index = self.load_vocab_glove_index()
        self.title_index_dict, self.abstract_index_dict = self.get_title_index_dict(), self.get_abstract_index_dict()

    def load_vocab_glove_index(self):
        """
        load glove vocab dictionary: word -> index
        :return:
        """
        glove_path = self.hyperParams["glove_path"]+"/6B."+str(self.hyperParams["model"]["embedding_size"])+"_idx.pkl"
        return pickle.load(open(glove_path, 'rb'))

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
                padding = [0 for i in range(maxLen - len(title_idx))]
                title_idx = title_idx + padding
            abstract_dict[id] = title_idx
        return abstract_dict

    def _word2idx(self, text):
        try:
            index = self.glove_vocab_index[text]
        except:
            index = 0
        return index

    def init_user_behavior(self):
        """
        Here we should address the problem of unbalanced data. (Copying positive data for five times.)
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
        return title_token, abstract_token

    def __len__(self):
        return len(self.behavior)

    def __getitem__(self, item):
        """
        Core function in torch Dataset. 20% of chances to take positive samples.
        :param item: index
        :return: clicked titles, clicked abstracts, candidate titles, candidate abstracts, candidate labels
        """
        candidate_titles, candidate_abstracts = self.get_candidates(item)
        candidate_labels = self.get_candidates_label(candidate_titles)
        # shuffle the samples
        # tmp = list(zip(candidate_titles, candidate_labels))
        try:
            tmp = list(zip(zip(candidate_titles, candidate_abstracts), candidate_labels))
            random.shuffle(tmp)
            candidate, candidate_labels = zip(*tmp)
            candidate_titles, candidate_abstracts = zip(*candidate)
        except:
            return self.__getitem__((item + 1) % self.__len__())
        return torch.tensor(candidate_titles[0]), torch.tensor(candidate_abstracts[0]), \
               torch.tensor(candidate_labels[0])

    def get_click_news(self, idx):
        """
        Get news title and abstract that are clicked by users
        :param idx: dataset index
        :return: news clicked by users
        """
        news_len = self.hyperParams["data"]["maxLen"]
        pos_num = self.hyperParams["data"]["pos_num"]

        # get clicked news from user behavior
        try:
            click_news_title = [self.title_index_dict[id] for id in self.behavior[idx][1][:news_len]]
            click_news_abstract = [self.abstract_index_dict[id] for id in self.behavior[idx][1][:news_len]]
        except:
            click_news_title, click_news_abstract = [], []

        zero_padding_title = [0] * self.hyperParams["data"]["titleLen"]
        zero_padding_abstract = [0] * self.hyperParams["data"]["wordLen"]
        if len(click_news_title) < pos_num:
            padding_news_title = [zero_padding_title for _ in range(pos_num-len(click_news_title))]
            padding_news_abstract = [zero_padding_abstract for _ in range(pos_num - len(click_news_title))]
            click_news_title = click_news_title + padding_news_title
            click_news_abstract = click_news_abstract + padding_news_abstract
        return click_news_title, click_news_abstract

    def get_candidates(self, idx):
        """
        Get candidate news' titles and abstracts.
        :param idx: dataset index
        :return: candidate news
        """
        neg_num = self.hyperParams["data"]["neg_num"]
        pos_impression = self.behavior[idx][2]
        neg_impression = self.behavior[idx][3]
        pos_sampled = random.sample(pos_impression, 1)
        try:
            neg_sampled = random.sample(neg_impression, neg_num)
        except:
            neg_sampled = neg_impression
            random.shuffle(neg_sampled)
        candidate_titles, candidate_abstracts = [], []
        sampled_id = pos_sampled + neg_sampled
        for id in sampled_id:
            try:
                candidate_titles.append(self.title_index_dict[id])
                candidate_abstracts.append(self.abstract_index_dict[id])
            except:
                continue

        return candidate_titles, candidate_abstracts

    def get_candidates_label(self, candidate_list):
        """
        Get candidate news labels.
        :return: Pos/Neg labels of samples
        """
        label = [1] + [0] * (len(candidate_list) - 1)
        return label


class NewsDatasetDCNVal(data.Dataset):
    def __init__(self, hyperParams, file_path):
        super(NewsDatasetDCNVal, self).__init__()
        self.hyperParams = hyperParams
        self.file_path = file_path
        self.behavior, self.clickHis = self.init_user_behavior()
        self.title_token, self.abstract_token = self.get_news()
        self.glove_vocab_index = self.load_vocab_glove_index()
        self.title_index_dict, self.abstract_index_dict = self.get_title_index_dict(), self.get_abstract_index_dict()

    def load_vocab_glove_index(self):
        """
        load glove vocab dictionary: word -> index
        :return:
        """
        glove_path = self.hyperParams["glove_path"]+"/6B."+str(self.hyperParams["model"]["embedding_size"])+"_idx.pkl"
        return pickle.load(open(glove_path, 'rb'))

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
                padding = [0 for i in range(maxLen - len(title_idx))]
                title_idx = title_idx + padding
            abstract_dict[id] = title_idx
        return abstract_dict

    def _word2idx(self, text):
        try:
            index = self.glove_vocab_index[text]
        except:
            index = 0
        return index

    def init_user_behavior(self):
        """
        Here we should address the problem of unbalanced data. (Copying positive data for five times.)
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
        return title_token, abstract_token

    def __len__(self):
        return len(self.behavior)

    def __getitem__(self, item):
        """
        Core function in torch Dataset. 20% of chances to take positive samples.
        :param item: index
        :return: clicked titles, clicked abstracts, candidate titles, candidate abstracts, candidate labels
        """
        candidate_titles, candidate_abstracts, pos_num, neg_num = self.get_candidates(item)
        candidate_labels = self.get_candidates_label(pos_num, neg_num)
        try:
            tmp = list(zip(zip(candidate_titles, candidate_abstracts), candidate_labels))
            random.shuffle(tmp)
            candidate, candidate_labels = zip(*tmp)
            candidate_titles, candidate_abstracts = zip(*candidate)
        except:
            return self.__getitem__((item + 1) % self.__len__())
        return torch.tensor(candidate_titles), torch.tensor(candidate_abstracts), \
               torch.tensor(candidate_labels)

    def get_click_news(self, idx):
        """
        Get news title and abstract that are clicked by users
        :param idx: dataset index
        :return: news clicked by users
        """
        news_len = self.hyperParams["data"]["maxLen"]
        pos_num = self.hyperParams["data"]["pos_num"]

        # get clicked news from user behavior
        try:
            click_news_title = [self.title_index_dict[id] for id in self.behavior[idx][1][:news_len]]
            click_news_abstract = [self.abstract_index_dict[id] for id in self.behavior[idx][1][:news_len]]
        except:
            click_news_title, click_news_abstract = [], []

        zero_padding_title = [0] * self.hyperParams["data"]["titleLen"]
        zero_padding_abstract = [0] * self.hyperParams["data"]["wordLen"]
        if len(click_news_title) < pos_num:
            padding_news_title = [zero_padding_title for _ in range(pos_num-len(click_news_title))]
            padding_news_abstract = [zero_padding_abstract for _ in range(pos_num - len(click_news_title))]
            click_news_title = click_news_title + padding_news_title
            click_news_abstract = click_news_abstract + padding_news_abstract
        return click_news_title, click_news_abstract

    def get_candidates(self, idx):
        """
        Get candidate news' titles and abstracts.
        :param idx: dataset index
        :return: candidate news
        """
        pos_impression = self.behavior[idx][2]
        neg_impression = self.behavior[idx][3]
        candidate_titles, candidate_abstracts = [], []
        sampled_id = pos_impression + neg_impression
        if len(pos_impression) == 0 or len(neg_impression) == 0:
            return self.get_candidates((idx+1) % self.__len__())
        for id in sampled_id:
            try:
                candidate_titles.append(self.title_index_dict[id])
                candidate_abstracts.append(self.abstract_index_dict[id])
            except:
                continue

        return candidate_titles, candidate_abstracts, len(pos_impression), len(neg_impression)

    def get_candidates_label(self, pos_num, neg_num):
        """
        Get candidate news labels.
        :return: Pos/Neg labels of samples
        """
        label = [1] * pos_num + [0] * neg_num
        return label


if __name__ == "__main__":
    from config import hyperParams
    hyperParams['glove_path'] = '../data/glove'
    hyperParams['train_data_path'] = '../data/train'
    hyperParams['val_data_path'] = '../data/val'
    hyperParams['test_data_path'] = '../data/val'
    train_set = NewsDatasetDCN(hyperParams, "../data/val")
    count = 10
    for batch in train_set:
        print(batch[0])
        print(batch[1])
        print(batch[2])
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print("---------------------------")
        if count == 0:
            break
        count -= 1
