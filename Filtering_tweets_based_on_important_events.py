"""
The objective of this code is to divide tweets in several files for topic modeling based on events about scandal
In this script also hashtags of the whole document will be searched for hashtags
"""
import pandas as pd
import numpy as np
import time
import csv
import datetime
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt

# function to read the csv file and gives a dataframe
def csv_to_dataframe(file_name):
    """
    Read a csv file and create a dataframe

    Parameters:
    ----------
    file_name : csv file name

    Returns:
    -------
    df : dataframe for that csv file
    """
    df=pd.read_csv(file_name , encoding="utf-8",delimiter=',',quotechar='"')
    return df


def find_hashtags_from_list(hashtags):
    """
    Create a dictionary of hashtags, The key is hashtag and value is how many times that hashtag occured

    Parameters:
    ----------
    hashtagsa : list of hashtags


    Returns:
    -------
    hash_dict : Dictionary of hashtags and their number of occurance
    """
    hash_dict={}
    for hashtags in hashtags.to_list():
        for word in hashtags:
            try:
                hash_dict[word]+=1
            except KeyError:
                hash_dict[word]=1
    return hash_dict
def hashtag_finder(tweet):
    """
    find all hashtags in tweets

    Parameters:
    ----------
    tweet : every tweet

    Returns:
    -------
    hashtags : List of hashtags in a tweet

    """
    hashtags = re.findall("#([a-zA-Z0-9_]{1,50})", tweet)
    return hashtags

def wordcloud(hashtag_dic):
    """
    Create a wordcloud based on dictionary of hashtags
    Parameters:
    ----------
    hashtag_dict : Dictionary of Hashtags
    Returns:
    -------
    image : image object of wordcloud created for hashtags
    """
    cloud = WordCloud(width=1200, height=1000).generate_from_frequencies(hashtag_dic)
    image = cloud.to_image()
    return image

def sentiment_daily_count(df):
    """
    Create a dataframe for finding unweighted sentiment gap on daily basis

    Parameters:
    ----------
    df : dataframe of all tweets and their sentiments

    Returns:
    -------
    df1 : dataframe grouped by on each day to see the sum of sentiments on each day

    """
    columns = ['created_at', 'sentiment']
    df1 = pd.DataFrame(df, columns=columns)
    return df1.groupby(['created_at','sentiment']).size()

def sum_sentiment_daily(df):
    """
    Create a dataframe for finding weighted sentiment gap on daily basis

    Parameters:
    ----------
    df : dataframe of all tweets and their sentiments

    Returns:
    -------
    df1 : dataframe grouped by on each day to see the sum of sentiments on each day
    
    """
    columns = ['created_at', 'sentiment_labled_int','weighted_sentiment']
    df1 = pd.DataFrame(df, columns=columns)
    return df1.groupby('created_at').agg( number_of_tweets=pd.NamedAgg(column='created_at',aggfunc='count'),
    weighted_sentiments=pd.NamedAgg(column='weighted_sentiment', aggfunc=sum),
    unweighted_sentiments=pd.NamedAgg(column='sentiment_labled_int', aggfunc=sum))

if __name__ == "__main__":

    file_name="tweets_labeled1.csv"
    df=csv_to_dataframe(file_name=file_name)
    df.describe()
    # Change date from isoformat to desired format
    df['created_at']= df['created_at'].apply(lambda x: datetime.datetime.fromisoformat(x).strftime('%Y-%m-%d'))
    # use the hashtag finder to put a column of hashtags for each tweet
    df['hashtags']= df['text'].apply(lambda x: hashtag_finder(x))
    hash_dict=find_hashtags_from_list(df['hashtags'])
    # create weighted sentiment for each tweet
    df['weighted_sentiment']=df['sentiment_labled_int']*df['followers']
    df.describe()
    columns = ['created_at','weighted_sentiment']
    # Create sentiment weighted and unweighted 
    sentiment_count=sentiment_daily_count(df)
    count_weighted_unweighted_sentiment=sum_sentiment_daily(df)
    
    # Use hashtag column to create wordcloud and create a datafram of hashtags and their frequency
    Hashtags = list(hash_dict.keys())
    Hashtag_counts = list(hash_dict.values())
    df_hashtag = pd.DataFrame({'Hashtag' : Hashtags,'Frequency' : Hashtag_counts})
    df_sorted_hashtag = df_hashtag.sort_values(by='Frequency', ascending=False)
    df_sorted_hashtag.to_csv('tweets_labeled1_hashtag_sorted.csv',encoding='utf-8')
    cloud=wordcloud(hash_dict)
    cloud.show()
    cloud.save("wordcloud_all_tweets.png")



    # Divide tweets into several files for topic modeling
    df_15thJanto29thJan=df[df['created_at'].between('2018-01-15', '2018-01-29')]
    df_09thFebto10thFeb=df[df['created_at'].between('2018-02-09', '2018-02-10')]
    df_11thFebto17thFeb=df[df['created_at'].between('2018-02-11', '2018-02-17')]
    df_18thFebto19thFeb=df[df['created_at'].between('2018-02-18', '2018-02-19')]
    df_20thFebto28thFeb=df[df['created_at'].between('2018-02-20', '2018-02-28')]
    df_29thFebto09thMay=df[df['created_at'].between('2018-02-29', '2018-05-09')]
    df_10thMayto20thMay=df[df['created_at'].between('2018-05-10', '2018-05-20')]

    df.to_csv('tweets_labeled_with_sentiment_wighted.csv', encoding='utf-8')
    df_15thJanto29thJan.to_csv('tweets_labeled_15thJanto29thJan.csv', encoding='utf-8')
    df_09thFebto10thFeb.to_csv('tweets_labeled_9thFebto10thFeb.csv', encoding='utf-8')
    df_11thFebto17thFeb.to_csv('tweets_labeled_11thFebto17thFeb.csv', encoding='utf-8')
    df_18thFebto19thFeb.to_csv('tweets_labeled_18thFebto19thFeb.csv', encoding='utf-8')
    df_20thFebto28thFeb.to_csv('tweets_labeled_20thFebto28thFeb.csv', encoding='utf-8')
    df_29thFebto09thMay.to_csv('tweets_labeled_29thFebto09thMay.csv', encoding='utf-8')
    df_10thMayto20thMay.to_csv('tweets_labeled_10thMayto20thMay.csv', encoding='utf-8')
    sentiment_count.to_csv('sentiment_count_daily.csv', encoding='utf-8')
    count_weighted_unweighted_sentiment.to_csv('wieghed_and_unweighted_sentiment_gap_daily.csv', encoding='utf-8')









