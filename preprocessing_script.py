import pandas as pd
import numpy as np

def import_news_data():
    print('Importing news data...')
    news_data = pd.read_csv('./dataset/news.tsv', header=None, sep='\t')
    news_data.columns = ["News ID",
                        "Category",
                        "SubCategory",
                        "Title",
                        "Abstract",
                        "URL",
                        "Title Entities",
                        "Abstract Entities"] 
    news_data.dropna(inplace=True)
    news_data.reset_index(drop=True,inplace=True)
    return news_data

def import_behaviors_data(news_data):
    print('Importing behaviors data...')
    behaviors_data = pd.read_csv('./dataset/behaviors.tsv', header=None, sep='\t')
    behaviors_data.columns = ["Impression ID",
                        "User ID",
                        "Time",
                        "History",
                        "Impressions"]
    behaviors_data.dropna(inplace=True)
    behaviors_data.reset_index(drop=True,inplace=True)

    # count the number of history of each user
    len_history = []
    for i in range(len(behaviors_data)):
        len_history.append(len(behaviors_data['History'][i].split(' ')))
    behaviors_data['Number of history'] = len_history

    # Category and Sub Category in history
    category_history = []
    sub_category_history = []
    for i in range(len(behaviors_data)):
        history_list = behaviors_data['History'][i].split(' ')
        temp = ' '.join(news_data[news_data['News ID'].isin(history_list)]['Category'])
        category_history.append(temp)

        temp = ' '.join(news_data[news_data['News ID'].isin(history_list)]['SubCategory'])
        sub_category_history.append(temp)

        if i%10000==0:
            print(i)

    behaviors_data['History category']=category_history
    behaviors_data['History subcategory']=sub_category_history

    # Time category
    time = list()
    for i in range(len(behaviors_data)):
        hour = int(behaviors_data['Time'][i].split(' ')[1].split(':')[0])
        ap = behaviors_data['Time'][i].split(' ')[2]
        if 6 <= hour < 12 and ap == 'AM': # Morning
            t = 'Morning'
        if 0 <= hour < 6 and ap == 'PM': # Afternoon
            t = 'Afternoon'
        if 6 <= hour < 12 and ap == 'PM': # Evening
            t = 'Evening'
        if 0 <= hour < 6 and ap == 'AM': # Night
            t = 'Night' 
        time.append(t)
    behaviors_data['Time category'] = time

    # Impression rate
    impressions = dict()
    for i in range(len(behaviors_data)):
        user_id = behaviors_data['User ID'][i]
        temp = ('-'.join(behaviors_data['Impressions'][i].split(' '))).split('-')[1::2]
        i_list = list(map(int, temp))
        if user_id in impressions.keys():
            impressions[user_id] = impressions[user_id] + i_list
        else:
            impressions[user_id] = i_list
            
    impressions_rate = list()
    for i in range(len(behaviors_data)):
        user_id = behaviors_data['User ID'][i]
        impressions_rate.append(sum(impressions[user_id])/len(impressions[user_id]))
    behaviors_data['Impressions_rate'] = impressions_rate
    return behaviors_data

def export_behaviors_data(behaviors_data):
    print('Exporting new data...')
    # save processed data to behaviors1.csv
    behaviors_data.to_csv('behaviors1.csv', index=False, header=True)

if __name__ == '__main__':
    news_data = import_news_data()
    behaviors_data = import_behaviors_data(news_data)
    export_behaviors_data(behaviors_data)
    