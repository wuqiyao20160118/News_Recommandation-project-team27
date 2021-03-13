import pandas as pd
from utils.utils import tokenize_word
import random
import pickle
import torch
from torch.utils import data


class NewsDataset(data.Dataset):
    def __init__(self, hyperParams, file_path):
        super(NewsDataset, self).__init__()
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
        Core function in torch Dataset. Get mini batches from the data by index.
        :param item: index
        :return: clicked titles, clicked abstracts, candidate titles, candidate abstracts, candidate labels
        """
        click_titles, click_abstracts = self.get_click_news(item)
        candidate_titles, candidate_abstracts = self.get_candidates(item)
        candidate_labels = self.get_candidates_label()
        # shuffle the samples
        # tmp = list(zip(candidate_titles, candidate_labels))
        tmp = list(zip(candidate_abstracts, candidate_labels))
        random.shuffle(tmp)
        candidate_abstracts, candidate_labels = zip(*tmp)
        return torch.tensor(click_abstracts), torch.tensor(candidate_abstracts), torch.tensor(candidate_labels)

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

        zero_padding = [0] * self.hyperParams["data"]["wordLen"]
        if len(click_news_title) < pos_num:
            padding_news = [zero_padding for _ in range(pos_num-len(click_news_title))]
            click_news_title = click_news_title + padding_news
            click_news_abstract = click_news_abstract + padding_news
        return click_news_title, click_news_abstract

    def get_candidates(self, idx):
        """
        Get candidate news' titles and abstracts.
        :param idx: dataset index
        :return: candidate news
        """
        news_len = self.hyperParams["data"]["wordLen"]
        neg_num = self.hyperParams["data"]["neg_num"]
        pos_impression = self.behavior[idx][2]
        neg_impression = self.behavior[idx][3]
        try:
            neg_sampled = random.sample(neg_impression, neg_num)
        except:
            neg_sampled = neg_impression
            random.shuffle(neg_sampled)
        candidate_titles, candidate_abstracts = [], []
        sampled_id = pos_impression + neg_sampled
        for id in sampled_id:
            try:
                candidate_titles.append(self.title_index_dict[id])
                candidate_abstracts.append(self.abstract_index_dict[id])
            except:
                continue

        zero_padding = [0] * news_len
        if len(candidate_titles) <= neg_num:
            padding_news = [zero_padding for _ in range(neg_num + 1 - len(candidate_titles))]
            candidate_titles = candidate_titles + padding_news
            candidate_abstracts = candidate_abstracts + padding_news

        return candidate_titles, candidate_abstracts

    def get_candidates_label(self):
        """
        Get candidate news labels.
        :return: Pos/Neg labels of samples
        """
        label = [1] + [0] * self.hyperParams["data"]["neg_num"]
        return label


if __name__ == "__main__":
    from config import hyperParams
    train_set = NewsDataset(hyperParams, "../data/val")
    count = 10
    for batch in train_set:
        print(batch[2])
        print(batch[4])
        print("---------------------------")
        if count == 0:
            break
        count -= 1
