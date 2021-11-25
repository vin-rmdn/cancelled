import twitter
import pandas
import numpy
import re

class TweetFetch():
    def __init__(self, bearer_file='../credential/bearer_token.txt') -> None:
        # Initialize Twitter
        with open(bearer_file) as f:
            self.BEARER_TOKEN = f.read()
            print('Token loaded!')
        # print('Token:', self.BEARER_TOKEN)
        self.twitter_handle = twitter.Twitter(auth=twitter.OAuth2(bearer_token=self.BEARER_TOKEN))

    def get_tweet_raw(self, q, lang='en', max_tweets=1000):
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

            query['inferred_text'] = query_text

            print(len(query_text), end='')

            # Saving to Tweet Data Frame
            tweet_df = tweet_df.append(query).reset_index().drop('index', axis=1) if tweet_df is not None else query
            iteration += 1
            print(' =', len(tweet_df), end='\n')
            if len(tweet_df) >= max_tweets:
                # print(tweet_df.head())
                tweet_df = tweet_df.drop_duplicates(subset=['inferred_text']).reset_index().drop('index', axis=1)
        return tweet_df
