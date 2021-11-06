import twitter
import pandas
import re
import tensorflow
import pickle
import numpy

tensorflow.get_logger().setLevel('INFO')

class TweetFetch():
    def __init__(self, bearer_file) -> None:
        # Initialize Twitter
        with open(bearer_file) as f:
            self.BEARER_TOKEN = f.read()
        print('Token:', self.BEARER_TOKEN)
        self.twitter_handle = twitter.Twitter(auth=twitter.OAuth2(bearer_token=self.BEARER_TOKEN))

        # Initialize Neural Network and Tokenizer
        self.model = tensorflow.keras.models.load_model('../model/twitter-spam')
        with open('../dataset/squid-tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Metadata           

    def preprocess_text(self, input):
        input.replace('(\@[A-Za-z0-9]+)', '<username>', regex=True, inplace=True) # Username obfuscation
        input.replace('http[^ ]+', '<link>', regex=True, inplace=True) # Link obfuscation
        input.replace('(\#[A-Za-z]+)', '<hashtag>', regex=True, inplace=True) # Hashtag obfuscation

    def predict_spam(self, series):
        df_input = series.copy()
        self.preprocess_text(df_input)
        sequence = self.tokenizer.texts_to_sequences(df_input)
        pad_seq = tensorflow.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=75)
        return self.model.predict(pad_seq)

    def filter_spam(self, df):
        df['is-spam'] = self.predict_spam(df['tweet'])

    def get_tweet(self, q, lang='en', max_tweets=3000):
        tweet_df = pandas.DataFrame()
        since_id_buff = ''
        iteration = 0
        while len(tweet_df) < max_tweets:
            print('Iteration', iteration + 1, end=': ')
            # Fetch from Twitter
            query = self.twitter_handle.search.tweets(q=q, lang=lang, count=100, tweet_mode='extended') if since_id_buff == '' else self.twitter_handle.search.tweets(q=q, lang=lang, count=100, tweet_mode='extended', max_id=since_id_buff)

            since_id_buff = re.search(r'max_id=(.+)&q', query['search_metadata']['next_results']).group(1)  # get last max_id
            query = pandas.DataFrame(query['statuses'])

            # print(len(query))
            # Getting only fulltext
            query_rt = []
            for i in query['retweeted_status']:
                try:
                    query_rt.append(i['full_text'])
                except:
                    query_rt.append(numpy.nan)
            # print(query_rt)
            query_text = []
            for i in range(len(query_rt)):
                # print(i, end='\r')
                query_text.append(query_rt[i] if not pandas.isnull(query_rt[i]) else query['full_text'].loc[i])
            
            print(len(query_text), end=' -> ')
            # Filtering spam
            query_text = pandas.DataFrame(query_text)
            query_text.columns = ['tweet']
            self.filter_spam(query_text)

            # Drop spam
            # query_text.drop(query_text[query_text['is-spam'] > 0.5].index, inplace=True)
        
            # print(query_text)
            print(len(query_text), end='')
            # Saving to Tweet Data Frame
            tweet_df = tweet_df.append(query_text).reset_index().drop('index', axis=1) if tweet_df is not None else query_text
            iteration += 1
            print(' =', len(tweet_df), end='\n')
            if len(tweet_df) >= max_tweets:
                tweet_df = tweet_df.drop_duplicates().reset_index().drop('index', axis=1)
        return tweet_df