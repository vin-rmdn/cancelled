import pickle
import sklearn
import re
import string
import emoji
import nltk
import pandas as pd
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TweetSpamPredictor:

    # Functions
    def __init__(self):
        print('Loading models', end='...')
        with open('../model/model_rf.pickle', 'rb') as f:
            self.model = pickle.load(f)
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.lemmatized_tweets = pd.read_json('../dataset/tweet-processed.json', orient='index')['lemmatized_text']
        print('Done!\nCreating TF-IDF vectorizer', end='...')
        self.tfidfvectorizer = TfidfVectorizer(analyzer=lambda x: x).fit(self.lemmatized_tweets.values)
        print('Done!\nLoading Word2Vec vectorizer', end='...')
        self.word2vec = gensim.models.Word2Vec.load('../model/tweets.embedding')
        print('Done!\nCreating Word2Vec vectorizer', end='...')
        embedding_matrix = np.zeros((len(self.tfidfvectorizer.vocabulary_), 50))
        print('Done!\nCreating embedded vectorizer', end='...')
        self.embedding_matrix = np.zeros((len(self.tfidfvectorizer.vocabulary_), 50))
        for word, i in self.tfidfvectorizer.vocabulary_.items():
            if word in self.word2vec.wv.index_to_key:
                self.embedding_matrix[i] = self.word2vec.wv[word]

    def preprocess_text(self, text):
        return_vec = re.sub('(\@[A-Za-z0-9_]+)', '', text)   # Username removal
        return_vec = re.sub(r'([http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))', '', return_vec)   # Link removal
        return_vec = re.sub('(#)', ' ', return_vec)  # Hashtag removal, but preserving the word
        return_vec = re.sub('([-&])', ' ', return_vec)   # dash spacing, accommodating for better representation in spams
        return_vec = "".join([i for i in return_vec if i not in string.punctuation])
        return_vec = re.sub('([’])', "'", return_vec)
        return_vec = re.sub(emoji.get_emoji_regexp(), r"", return_vec)
        return_vec = nltk.tokenize.word_tokenize(return_vec)
        return_vec = [i for i in return_vec if i not in self.stopwords]
        return_vec = [self.lemmatizer.lemmatize(i) for i in return_vec]
        return_vec = self.tfidfvectorizer.transform(return_vec)     # 1 x nWords
        return_vec = return_vec.toarray() @ self.embedding_matrix
        return pd.DataFrame(return_vec)

    def predict_spam(self, input_df):
        # Variables
        OPEN_CANDIDATE = [
            'entities',
            'metadata',
            'user'
        ]
        DROP_CAND = [
            'created_at',
            'id',
            'id_str',
            'full_text',
            'truncated',
            'source',
            'lang',
            'quoted_status_id_str',
            'metadata_iso_language_code',
            'metadata_result_type',
            'user_entities',
            'retweeted_status',
            'in_reply_to_status_id_str', #dup
            'in_reply_to_user_id_str', #dup
            'in_reply_to_screen_name', #redundant
            'in_reply_to_user_id',
            'user_id',
            'user_id_str',
            'user_screen_name',
            'user_url',
            'user_utc_offset',
            'user_time_zone',
            'user_lang',
            'user_profile_background_image_url',
            'user_profile_background_image_url_https',
            'user_profile_image_url',
            'user_profile_image_url_https',
            'user_profile_banner_url',
            'geo',
            'coordinates',
            'contributors',
            'place',
            'user_withheld_in_countries',
            'user_following',
            'user_follow_request_sent',
            'user_notifications',
            'extended_entities'
        ]

        # Saving a local instance
        tweet_df = input_df.copy()

        # print(OPEN_CANDIDATE)

        # Opening nested dataframe
        for attr in OPEN_CANDIDATE:
            # print(attr)
            attr_opened = pd.Series(tweet_df.get(attr)).add_prefix(attr+'_')
            tweet_df = pd.concat([tweet_df.drop(attr), attr_opened])
            # print(tweet_df.index)

        # Dropping unused attr
        # print(tweet_df.index)
        tweet_df.drop(DROP_CAND, inplace=True)

        # print(tweet_df.index)

        # Simple transformations
        print('Simple Transformations')
        tweet_df['display_text_range'] = tweet_df['display_text_range'][1] - tweet_df['display_text_range'][0]
        tweet_df['is_replying_to_others'] = 1.0 if not np.isnan(tweet_df['in_reply_to_status_id']) else 0.0
        tweet_df['is_quoting_status'] = 1.0 if not np.isnan(tweet_df['quoted_status_id']) else 0.0
        tweet_df['hashtag_count'] = len(tweet_df['entities_hashtags'])
        tweet_df['user_mention_count'] = len(tweet_df['entities_user_mentions'])
        tweet_df['media_count'] = 0 if 'entities_media' not in tweet_df.index else 0 if \
            pd.isnull(tweet_df['entities_media']) else len(tweet_df['entities_media'])
        tweet_df['has_symbols'] = 1.0 if bool(tweet_df['entities_symbols']) else 0.0
        tweet_df['has_url'] = 1.0 if bool(tweet_df['entities_urls']) else 0.0
        tweet_df['user_is_regular_translator'] = 1.0 if tweet_df['user_translator_type'] == 'regular' else 0.0

        # Drop previous attr
        print('Dropping attributes')
        tweet_df.drop(['display_text_range', 'in_reply_to_status_id', 'quoted_status_id', 'entities_hashtags',
                       'entities_user_mentions', 'entities_symbols', 'entities_urls', 'quoted_status',
                       'favorited', 'retweeted', 'user_protected', 'user_contributors_enabled', 'user_is_translator',
                       'user_translator_type'], inplace=True)
        if 'entities_media' in tweet_df.index:
            tweet_df.drop('entities_media')

        print('Doing several more calculations')
        tweet_df['user_profile_background_color_r'] = int(tweet_df['user_profile_background_color'][0:2], 16)
        tweet_df['user_profile_background_color_g'] = int(tweet_df['user_profile_background_color'][2:4], 16)
        tweet_df['user_profile_background_color_b'] = int(tweet_df['user_profile_background_color'][4:6], 16)
        tweet_df['user_profile_link_color_r'] = int(tweet_df['user_profile_link_color'][0:2], 16)
        tweet_df['user_profile_link_color_g'] = int(tweet_df['user_profile_link_color'][2:4], 16)
        tweet_df['user_profile_link_color_b'] = int(tweet_df['user_profile_link_color'][4:6], 16)
        tweet_df['user_profile_sidebar_border_color_r'] = int(tweet_df['user_profile_sidebar_border_color'][0:2], 16)
        tweet_df['user_profile_sidebar_border_color_g'] = int(tweet_df['user_profile_sidebar_border_color'][2:4], 16)
        tweet_df['user_profile_sidebar_border_color_b'] = int(tweet_df['user_profile_sidebar_border_color'][4:6], 16)
        tweet_df['user_profile_sidebar_fill_color_r'] = int(tweet_df['user_profile_sidebar_fill_color'][0:2], 16)
        tweet_df['user_profile_sidebar_fill_color_g'] = int(tweet_df['user_profile_sidebar_fill_color'][2:4], 16)
        tweet_df['user_profile_sidebar_fill_color_b'] = int(tweet_df['user_profile_sidebar_fill_color'][4:6], 16)
        tweet_df['user_profile_text_color_r'] = int(tweet_df['user_profile_text_color'][0:2], 16)
        tweet_df['user_profile_text_color_g'] = int(tweet_df['user_profile_text_color'][2:4], 16)
        tweet_df['user_profile_text_color_b'] = int(tweet_df['user_profile_text_color'][4:6], 16)

        print('Getting vecs')
        vec = self.preprocess_text(tweet_df['inferred_text'])

        print('Dropping columns again')
        tweet_df.drop(['user_profile_background_color', 'user_profile_link_color', 'user_profile_sidebar_border_color',
                       'user_profile_sidebar_fill_color', 'user_profile_text_color'], inplace=True)

        # tweets = tweet_df[['is-spam', 'inferred_text']].rename({'inferred_text': 'raw_text'}, axis=1)

        tweet_df.drop(['inferred_text', 'user_name', 'user_location', 'user_description', 'user_created_at',
                       'possibly_sensitive'], inplace=True)

        print('Concatenating...')
        tweet_df = pd.concat([tweet_df, vec])

        print(tweet_df)

        return self.model.predict(tweet_df)






