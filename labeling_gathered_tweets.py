"""
The objective of this code is to clean tweets and do the sentiment analysis on them
"""
import pandas as pd
import re
import sys
import nltk 
nltk.download(['stopwords','wordnet','omw-1.4','names','twitter_samples','averaged_perceptron_tagger','vader_lexicon','punkt'])
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification






# function to read the csv file and gives a dataframe
def csv_to_dataframe(file_name):

  df=pd.read_csv(file_name , encoding="utf-8",delimiter=',',quotechar='"')
  return df
#setting stop words to english for text proccessing
stop_words = set(stopwords.words('english'))

# cleaning proccess of the tweets
def clean_tweet(tweet):
  return tweet.strip()
def lowercase_tweet(tweet):
  return tweet.lower()
def remove_multi_punctuation(tweet):
  return re.sub('^\w\s+',"",tweet)
def remove_http_links(tweet):
  words_keep=[]
  for word in tweet.split():
    word = word.strip()
    if word.startswith("http"):
      words_keep.append("")
    else:
      words_keep.append(word)
  return ' '.join(words_keep)  
def remove_stop_words(tweet):
  words_keep=[]
  for word in tweet.split():
    word=word.strip()
    if word not in stop_words:
      words_keep.append(word)
  return ' '.join(words_keep)
def remove_characnum(tweet):
  return re.sub("[^a-z0-9]"," ", tweet)
def remove_RT(tweet): 
  return re.sub(r'^RT[\s]+', '', tweet) 
def remove_hash(tweet):
  return re.sub(r'[\#+]'," ",tweet)  
def remove_handel(tweet):
  return re.sub(r'@[^ ]+', '', tweet) 
def remove_number(tweet):
  tweet = re.sub('([0-9]+)', '', str(tweet))
  return tweet
def lemitiz(tweet):
  lemitized_words=[]
  for word in tweet.split():
    word=Word(word).lemmatize()
    lemitized_words.append(word)
  return " ".join(lemitized_words)

#using all cleaning proccesses together
def processed_tweets(tweet):
  tweet = remove_hash(tweet)
  tweet = remove_handel(tweet)
  tweet = remove_http_links(tweet)
  tweet = clean_tweet(tweet)
  tweet = remove_RT(tweet)
  tweet = lowercase_tweet(tweet)
  tweet = remove_multi_punctuation(tweet)
  tweet = remove_characnum(tweet)
  tweet = remove_number(tweet)
  tweet = remove_stop_words(tweet)
  tweet = lemitiz(tweet)
  return tweet

#creating vader sentiment analyser
def sentiment_vader(tweet):
  # Create a SentimentIntensityAnalyzer object.
  sid_obj = SentimentIntensityAnalyzer()
  sentiment_dict = sid_obj.polarity_scores(tweet)
  if sentiment_dict['compound'] >= 0.05 :
      overall_sentiment = "Positive"
  elif sentiment_dict['compound'] <= -0.05:
      overall_sentiment = "Negative"
  else:
    overall_sentiment="Neutral"
  return overall_sentiment

#creating textblob sentiment analyser
def sentiment_texblob(tweet):
  classifier = TextBlob(tweet)
  polarity = classifier.sentiment.polarity
  if polarity> 0:
      sentiment = "Positive"
  elif polarity< 0:
      sentiment="Negative"
  else:
      sentiment="Neutral"
  return sentiment

##creating roberta pretrained sentiment analyser
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer_roberta = AutoTokenizer.from_pretrained(roberta)
classifier=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer_roberta)
def seentimet_roberta(tweet):
    sentiment=classifier(tweet)
    if sentiment[0]['label']=='LABEL_0':
      return 'Negative'
    elif sentiment[0]['label']=='LABEL_1':
      return 'Neutral'
    else:
      return 'Positive'

if __name__ == "__main__":
  file_name='tweets.csv'
  df=csv_to_dataframe(file_name)
  # Cleaning tweets
  df['cleaned_text']=df['text'].apply(processed_tweets)

  # Using sentiment anlysers on the tweets
  df['sentiment_vader']=df['cleaned_text'].apply(lambda x: sentiment_vader(x))
  df['sentiment_roberta']=df['text'].apply(lambda x: seentimet_roberta(x))
  df['sentiment_textblob']=df['cleaned_text'].apply(lambda x: sentiment_texblob(x))
  df.info()
  print(df['sentiment_textblob'].value_counts())
  print(df['sentiment_vader'].value_counts())
  print(df['sentiment_roberta'].value_counts())
  # create a majority voter for the final label of sentiments
  df['sentiment'] = df.iloc[:, 10:].mode(axis=1)[0]
  print(df['sentiment'].value_counts())
  df['sentiment_labled_int']=df['sentiment'].replace({"Negative": -1,"Neutral": 0,"Positive": 1 },inplace=False)
  df.to_csv('tweets_labeled1.csv', encoding='utf-8')

