"""
The objective of this code is to connect to twitter through twitter API and querry necessary data with relevant fields related to the Oxfam
"""
# Importing necessary packages
from time import sleep
import tweepy
import Config
import time
import re
import csv
import os
import pandas as pd
import datetime




def dic_builder(twitter_user_data):
    """
    Build a Dictionary of users information

    Parameters:
    ----------
    twitter_user_data : An object which twitter sends back after a call to get tweets. The object that Twitter sends back has three different entities and 
    includes entity is going to give us related information for users tweeted a post. (below more information will be given about twitter responds to make 
    it more clear)


    Returns:
    -------
    A dictionary which each user id is the key and for value there is another dictionary which user's username and followers is stored.
    """
    user_dict={}
    for user_info in twitter_user_data:
        user_dict[user_info.id] = {'username': user_info.username, 
        'followers': user_info.public_metrics['followers_count']}    
    return user_dict

def tweepy_paginator(method,query,user_fields,tweet_fields,expansions,start_time=None,end_time=None,max_result=500):
    """
    Getting tweets from twitter with Tweepy Paginator
    Tweepy paginitor makes searching tweets easier by requesting tweets in several requests to not go over the limits. In each request at most can get 500 tweets.
    It will store token related to that batch and then for the next request uses that token to get tweets after that token. Each request will bring back 3 objects,
    first is user fileds requested of the tweeters in that batch of tweets. Second object is the tweets and information asked to be retrived for each tweet, the 
    third object is the token of the first and last id of tweets and the number of results in taht batch.

    Parameters:
    ----------
    method : There is several methods for searching tweets, this program is going to access all of the database of twitter so searching all tweets method will be used
    query : String which is a query you want from twitter
    texts : List of input texts
    user_fields : List of user fields to be retrived.
    tweet_fields : List of tweet fields to be retrived. 
    expansions : This item allows to relate the tweets to additional information with users. (String)
    start_time : The start date(Isoformat date with zone included)
    end_time : The end date
    max_result : Max num of tweets in each request to be retrived
    Returns:
    -------
    tweeter_data : List of objects retrived
    
    """
    tweeter_data=tweepy.Paginator(method=method, query = query,
                                user_fields = user_fields,
                                tweet_fields = tweet_fields,
                                expansions = expansions,
                                start_time = start_time,
                                end_time = end_time,
                                max_results=max_result)
    return tweeter_data

                                
                                


def file_data_appeander(file_name,data_of_tweets,data_of_tweeters):
    """
    Write the tweets and user data to a file

    Parameters:
    ----------
    file_name : name of the file you want to write in it
    data_of_tweets : data of tweets gathered from paginator
    data_of_tweeters : dictionary created for each user


    Returns:
    -------
    File with tweets and users
    """
    with open(file=file_name,mode='a',encoding="utf-8") as f:
        for user in data_of_tweets:
            author_info=data_of_tweeters[user.author_id]
            row=[user.author_id,author_info['username'],author_info['followers'],
                re.sub("[\r\n]+"," ",user.text),user.created_at,user.public_metrics['retweet_count'],
                user.public_metrics['reply_count'],user.public_metrics['like_count'],user.public_metrics['quote_count']]
            csv_writer=csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
            csv_writer.writerow(row)

def find_last_end_time(df):
    """
    Finding the last tweet date

    Parameters:
    ----------
    df : dataframe of tweets

    Returns:
    -------
    date : date in the format acceptable for tweepy paginator
    """
    if df.empty:
        return 'created_at'
    else: 
        date=df.iloc[-1][4]
        date = datetime.datetime.fromisoformat(date).replace(tzinfo=datetime.timezone.utc)
        return date




def main():
    # Name of the file to store the details of tweets
    file_to_write='tweets.csv'
    # user fields requested from twitter for each tweet
    user_fields = ['username', 'public_metrics']
    # tweet fields requested from twitter for each tweet
    tweet_fields = ['created_at', 'public_metrics', 'text']
    # query all tweets contain oxfam and all the english ones
    query = 'oxfam lang:en'
    # the link between tweet data and user data for each tweet
    expansions = 'author_id'
    # start time for gathering tweets
    start_time = '2018-01-15T00:00:00Z'
    # end time for gathering tweets
    end_time = '2018-05-23T00:00:00Z'
    # To access the twitter data it is necessary to pass the baerer token rrecieved from twitter for research account
    client = tweepy.Client(bearer_token=Config.baerer_token, wait_on_rate_limit=True)
    try:
        # Create a try except if mean while of getting tweets the connection got disrupted when we rerun the file its gonna go and find the last tweet gathered 
        # date and put it as new end date for the requests.

        df=pd.read_csv(file_to_write , encoding="utf-8",delimiter=',',quotechar='"')
    except:
        df=pd.DataFrame()
    if  df.empty:
        with open(file='tweets.csv',mode='a',encoding="utf-8") as f:
            csv_writer=csv.writer(f,quoting=csv.QUOTE_NONNUMERIC)
            csv_writer.writerow(header)
        for tweeter_data in tweepy_paginator(method=client.search_all_tweets,
        query=query,user_fields= user_fields,tweet_fields= tweet_fields
        ,expansions= expansions,start_time= start_time,end_time= end_time):
            time.sleep(1)
            user_dic = dic_builder(tweeter_data.includes['users'])
            file_data_appeander(file_to_write,data_of_tweets=tweeter_data.data,data_of_tweeters=user_dic)
    else:
        last_date=find_last_end_time(df)
        for tweeter_data in tweepy_paginator(method=client.search_all_tweets,
        query=query,user_fields= user_fields,tweet_fields= tweet_fields
        ,expansions= expansions,start_time= start_time,end_time= last_date):
            time.sleep(1)
            user_dic = dic_builder(tweeter_data.includes['users'])
            file_data_appeander(file_to_write,data_of_tweets=tweeter_data.data,data_of_tweeters=user_dic)

if __name__ == "__main__":
    header=['author_id','username','followers','text','created_at','retweets','replies','likes','quote_counts']

    with open(file='tweets.csv',mode='a',encoding="utf-8") as f:
        pass
    main()
