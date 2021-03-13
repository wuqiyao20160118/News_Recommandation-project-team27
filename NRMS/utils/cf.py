import time, math, os
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
from collections import defaultdict
import collections
from tqdm import tqdm


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100*(start_mem-end_mem)/start_mem, (time.time()-starttime)/60))
    return df


def to_timestamp(df):
    t = df["Time"][:-3]
    timeArray = time.strptime(t, "%m/%d/%Y %H:%M:%S")
    timestamp = (int)(time.mktime(timeArray))
    return timestamp


def filter_impression(df):
    pos = []
    impressions = df["Impressions"]
    impressions = impressions.split(" ")
    for impression in impressions:
        impression_list = impression.split("-")
        if impression_list[1] == "1":
            pos.append(impression_list[0])
    return pos


def get_user_item_time(df):
    df = df.sort_values("Timestamp")
    
    def make_item_time_pair(df):
        return list(zip(df["Pos_click"], df["Timestamp"]))
    
    user_item_time_df = df.groupby("User_ID")["Pos_click", "Timestamp"].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: "item_time_list"})
    user_item_time_dict = dict(zip(user_item_time_df["User_ID"], user_item_time_df["item_time_list"]))
    return user_item_time_dict


def itemcf_sim(df, save_path):
    """
        Calculate the item-to-item similarity matrix
        :param df: dataframe
        return : item-to-item similarity matrix
        
    """
    
    user_item_time_dict = get_user_item_time(df)
    
    # Calculate the similarity
    item2item_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for item_list, i_click_time in item_time_list:
            for item in item_list: 
                item_cnt[item] += 1
                item2item_sim.setdefault(item, {})
                for another_item_list, _ in item_time_list:
                    for another_item in another_item_list:
                        if(item == another_item):
                            continue
                        item2item_sim[item].setdefault(another_item, 0)
                        item2item_sim[item][another_item] += 1 / math.log(len(item_time_list) + 1)
                
    item2item_sim_ = item2item_sim.copy()
    for item, related_items in item2item_sim.items():
        for related_item, wij in related_items.items():
            item2item_sim_[item][related_item] = wij / math.sqrt(item_cnt[item] * item_cnt[related_item])
    
    # save the data persistently
    pickle.dump(item2item_sim_, open(save_path + 'itemcf_item2item_sim.pkl', 'wb'))
    
    return item2item_sim_


def get_item_click_num(df):
    count_map = {}
    for i in range(df.shape[0]):
        pos_click = df.loc[i, "Pos_click"]
        for item_id in pos_click:
            count_map.setdefault(item_id, 0)
            count_map[item_id] += 1
    return count_map
        

def get_item_topk_click(count_map, k):
    topk_click = sorted(count_map.items(), key=lambda x: x[1], reverse=True)[:k]
    return [item_id for (item_id, _) in topk_click]


def load_data(hyperParams, stage="train"):
    assert stage in ["train", "val", "test"]
    file_path = hyperParams[stage+"_data_path"]
    file_path = file_path + "/behaviors.tsv"
    user_behavior = pd.read_csv(file_path, header=None, sep='\t')
    user_behavior.columns = ["Impress_ID",
                             "User_ID",
                             "Time",
                             "History",
                             "Impressions"]
    user_behavior["Pos_click"] = user_behavior.apply(lambda x: filter_impression(x), axis=1)
    user_behavior["Timestamp"] = user_behavior.apply(lambda x: to_timestamp(x), axis=1)
    return user_behavior


def get_user_label(df):
    user_label_dict = dict(zip(df["User_ID"], df["Pos_click"]))
    return user_label_dict


def get_news_category(hyperParams, stage="train"):
    assert stage in ["train", "val", "test"]
    file_path = hyperParams[stage + "_data_path"]
    path = file_path + "/news.tsv"
    news = pd.read_csv(path, header=None, sep='\t')
    news.columns = ["ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_entities",
                        "Abstract_entities"]
    id2category = {}
    for i in range(news.shape[0]):
        id, category =  news.loc[i, "ID"], news.loc[i, "Category"]
        if not isinstance(category, str):
            continue
        category = category.lower()
        id2category[id] = category
    return id2category


def merge_category_map(train_map, test_map):
    category_map = {}
    for k, v in train_map.items():
        category_map[k] = v
    for k, v in test_map.items():
        category_map[k] = v
    return category_map


def item_based_recommend(user_id, user_item_time_dict, item2item_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        recall based on itemCF
        :param user_id: user id
        :param user_item_time_dict: {user1: [(item1, time1), (item2, time2)..]...}
        :param item2item_sim: similarity matrix
        :param sim_item_topk: select top k similar news
        :param recall_item_num: the number of news be recalled
        :param item_topk_click: a list of top k news favored by all users        
        return: {item1: score1, item2: score2...}
    """
    
    # fetch the user's history clicks
    hist_items = user_item_time_dict[user_id]
    user_hist_items = []
    for (item_list, click_time) in hist_items:
        user_hist_items.extend(item_list)
    user_hist_items_ = {item_id for item_id in user_hist_items}
    
    item_rank = {}
    for item in user_hist_items:
        try:
            for another_item, wij in sorted(item2item_sim[item].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if another_item in user_hist_items_:
                    continue

                item_rank.setdefault(another_item, 0)
                item_rank[another_item] +=  wij
        except:
            continue
    
    # fill the item_rank if the number of news in item_rank is less than recall_item_num
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():
                continue
            item_rank[item] = - i - 100 # set a random negative number
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank


def itemcf(df, item_count_map, save_path):
    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(df)
    item2item_sim = pickle.load(open(save_path + 'itemcf_item2item_sim.pkl', 'rb'))
    sim_item_topk = 10
    recall_item_num = 10
    item_topk_click = get_item_topk_click(item_count_map, k=50)
    for user in tqdm(df['User_ID'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, item2item_sim, sim_item_topk, recall_item_num, item_topk_click)
    return user_recall_items_dict


def get_recall_df(user_recall_items_dict, category_map):
    user_item_score_list = []
    category_count = defaultdict(int)
    for user, items in tqdm(user_recall_items_dict.items()):
        item_list, category_list = [], []
        for item, _ in items:
            item_list.append(item)
            category_list.append(category_map[item])
            category_count[category_map[item]] += 1
        max_category, max_count = "", -1
        for k, v in category_count.items():
            if max_count < v:
                max_count = v
                max_category = k
        user_item_score_list.append([user, item_list, category_list, max_category])
    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'category', 'most category'])
    return recall_df


def get_label_df(user_label_dict, category_map):
    user_label_list = []
    category_count = defaultdict(int)
    for user, items in tqdm(user_label_dict.items()):
        category_list = []
        for item in items:
            category_list.append(category_map[item])
            category_count[category_map[item]] += 1
        max_category, max_count = "", -1
        for k, v in category_count.items():
            if max_count < v:
                max_count = v
                max_category = k
        user_label_list.append([user, items, category_list, max_category])
    label_df = pd.DataFrame(user_label_list, columns=['user_id', 'click_article_id', 'category', 'most category'])
    return label_df
