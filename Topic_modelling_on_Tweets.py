"""
The objective of this code is to find topics of conversation
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pprint import pprint
import re
import gensim
import gensim.models.ldamulticore
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import Filtering_tweets_based_on_important_events








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
def Sort(sub_li):

  """
    Sort a list of lists based on the second value of the inside list

    Parameters:
    ----------
    subli: list 

    Returns:
    -------
    sub_li : Ordered list based on the second value of the inside list
    
  """
  sub_li.sort(key = lambda x: x[1],reverse=True)
  return sub_li


def gen_words(tweet):
  """
    using gensim simple preprocess for tweets before toipic modeling to have a better output with more cleaner text

    Parameters:
    ----------
    tweet : text of tweets


    Returns:
    -------
    tweet : proccessed tweet 
  """
  tweet = gensim.utils.simple_preprocess(str(tweet), deacc=True)
  return tweet

def make_bigrams(tweets):
  """
    Creating biagrams from tweets 

    Parameters:
    ----------
    tweet : text of tweets

    Returns:
    -------
    list of biagrams : list of possible biagrams from the tweet

  """
  return([bigram[doc] for doc in tweets])

def make_trigrams(tweets):
  """
    Creating biagrams from tweets 

    Parameters:
    ----------
    tweet : text of tweets

    Returns:
    -------
    list of trigram : list of possible trigram from the tweet

  """
  return ([trigram[bigram[doc]] for doc in tweets])


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    optimal_model: The best model with the highest coherence score
    """
    coherence_values = []
    model_list = []
    optimal_model=[]
    for num_topics in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,random_state=100,
                                           update_every=1,chunksize=100,
                                           passes=10,alpha="auto")
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        optimal_model.append([lda_model,coherencemodel.get_coherence()])
        Sort(optimal_model)
    return model_list, coherence_values,optimal_model[0][0]

def format_topics_sentences(ldamodel, corpus, texts):
    """
    Put the topic of each tweet and the dominant words of that in a dataframe

    Parameters:
    ----------
    ldamodel : lda model that been created for the corpus
    corpus : Gensim corpus
    texts : List of input texts


    Returns:
    -------
    sent_topics_df : Dataframe with the topics of each corpus and top keywords of that topic

    """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

if __name__ == "__main__":
  file_names=['tweets_labeled_9thFebto10thFeb','tweets_labeled_11thFebto16thFeb','tweets_labeled_10thMayto20thMay','tweets_labeled_15thJanto29thJan',
  'tweets_labeled_18thFebto19thFeb','tweets_labeled_21thFebto28thFeb']
  for file_name in file_names:
    
    df=csv_to_dataframe(file_name=file_name+'.csv')
    # Use hashtags functions created to analyse hashtags of each file and create a wordcloud
    df['hashtags']= df['text'].apply(lambda x: Filtering_tweets_based_on_important_events.hashtag_finder(x))
    hash_dict=Filtering_tweets_based_on_important_events.find_hashtags_from_list(df['hashtags'])
    Hashtags = list(hash_dict.keys())
    Hashtag_counts = list(hash_dict.values())
    df_hashtag = pd.DataFrame({'Hashtag' : Hashtags,'Frequency' : Hashtag_counts})
    df_sorted_hashtag = df_hashtag.sort_values(by='Frequency', ascending=False)
    df_sorted_hashtag.to_csv(file_name+'_hashtag_sorted.csv',encoding='utf-8')
    cloud=Filtering_tweets_based_on_important_events.wordcloud(hash_dict)
    cloud.save(file_name+"wordcloud_hashtags.png")
    # Use gensim preproccessor to prepare the cleaned tweets for topic analysis
    df['gen_words']= df['cleaned_text'].apply(lambda x: gen_words(x))
    #BIGRAMS AND TRIGRAMS
    bigram_phrases = gensim.models.Phrases(df['gen_words'], min_count=5, threshold=100)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[df['gen_words']], threshold=100)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)


    data_bigrams = make_bigrams(df['gen_words'])
    data_bigrams_trigrams = make_trigrams(data_bigrams)
    # Create a dictionary of the documnets
    id2word = corpora.Dictionary(data_bigrams_trigrams)

    id2word.filter_extremes(no_below=2, no_above=.99)
    
    data_lemmatized=df['gen_words']
    corpus = [id2word.doc2bow(d) for d in df['gen_words']]
    # Use the function and find the best number of topics by comparing coherence values of each mmodel in a scatter plot
    model_list, coherence_values,optimal_model = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=12, step=1)
    limit=12; start=2; step=1;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(file_name+'_coherence.png')
    plt.clf()

    for m, cv in zip(x, coherence_values):
      print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    # Use the optimal model with the highest coherence value and see the most used word for each topics
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))
    # Get the topics in pyLDAvis visualised format
    vis=pyLDAvis.gensim_models.prepare(optimal_model, corpus, id2word, mds="mmds", R=30)
    pyLDAvis.save_html(data=vis,fileobj=file_name+'_pylda.html')
    # Create the dataframe for corpus and related topics
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data_lemmatized)  
    df_topic_sents_keywords.to_csv(file_name+'_topic_modeling.csv', encoding='utf-8')                                      
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.to_csv(file_name+'_topic_modeling_index_reset.csv', encoding='utf-8')  
    print(df_dominant_topic.head(10))
    # Change the dataframe and group by to see how many documnets is in different topic
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                               grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                              axis=0)
    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    print(sent_topics_sorteddf_mallet.head())
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
    df_dominant_topics.to_csv(file_name+'_topics.csv', encoding='utf-8')
