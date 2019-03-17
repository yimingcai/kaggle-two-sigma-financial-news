# leave only latest news
# flat subjects
# flat audiences
# binary bow headline

# ![](http://)![](http://)************# Two Sigma Financial News Competition Official Getting Started Kernel
# ## Introduction
# In this competition you will predict how stocks will change based on the market state and news articles.  You will loop through a long series of trading days; for each day, you'll receive an updated state of the market, and a series of news articles which were published since the last trading day, along with impacted stocks and sentiment analysis.  You'll use this information to predict whether each stock will have increased or decreased ten trading days into the future.  Once you make these predictions, you can move on to the next trading day. 
# 
# This competition is different from most Kaggle Competitions in that:
# * You can only submit from Kaggle Kernels, and you may not use other data sources, GPU, or internet access.
# * This is a **two-stage competition**.  In Stage One you can edit your Kernels and improve your model, where Public Leaderboard scores are based on their predictions relative to past market data.  At the beginning of Stage Two, your Kernels are locked, and we will re-run your Kernels over the next six months, scoring them based on their predictions relative to live data as those six months unfold.
# * You must use our custom **`kaggle.competitions.twosigmanews`** Python module.  The purpose of this module is to control the flow of information to ensure that you are not using future data to make predictions for the current trading day.
# 
# ## In this Starter Kernel, we'll show how to use the **`twosigmanews`** module to get the training data, get test features and make predictions, and write the submission file.
# ## TL;DR: End-to-End Usage Example
# ```
# from kaggle.competitions import twosigmanews
# env = twosigmanews.make_env()
# 
# (market_train_df, news_train_df) = env.get_training_data()
# train_my_model(market_train_df, news_train_df)
# 
# for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
#   predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
#   env.predict(predictions_df)
#   
# env.write_submission_file()
# ```
# Note that `train_my_model` and `make_my_predictions` are functions you need to write for the above example to work.

# In[1]:


import gc

import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!
from not_final_kernels.final_local_but_oom_kernel import parse_category_complex, flatten_category_complex

# In[ ]:

env = twosigmanews.make_env()

# ## **`get_training_data`** function
# 
# Returns the training data DataFrames as a tuple of:
# * `market_train_df`: DataFrame with market training data
# * `news_train_df`: DataFrame with news training data
# 
# These DataFrames contain all market and news data from February 2007 to December 2016.  See the [competition's Data tab](https://www.kaggle.com/c/two-sigma-financial-news/data) for more information on what columns are included in each DataFrame.

# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()

# In[ ]:


market_train_df.shape

# In[ ]:


market_train_df.head()

# In[ ]:


market_train_df.tail()

# In[ ]:


market_train_df.describe()

# In[ ]:


market_train_df.dtypes

# In[ ]:


market_train_df.volume.astype('float32').max()

# In[ ]:


market_train_df.volume.max()

# In[ ]:


news_train_df.head()

# In[ ]:


news_train_df.tail()


# In[ ]:


def add_id(df, id_name):
    df[id_name] = df.index.astype("int32") + 1


# ## compress dtypes

# In[ ]:


def compress_dtypes(news_df):
    for col, dtype in zip(news_df.columns, news_df.dtypes):
        if dtype == np.dtype('float64'):
            news_df[col] = news_df[col].astype("float32")
        if dtype == np.dtype('int64'):
            news_df[col] = news_df[col].astype("int32")


# In[ ]:


compress_dtypes(market_train_df)

# In[ ]:


compress_dtypes(news_train_df)

# In[ ]:


news_train_df.dtypes

# In[ ]:


news_train_df.tail()

# # add necessary info

# In[ ]:


MARKET_ID = "id"
NEWS_ID = "news_id"


# In[ ]:


def add_ids(market_df, news_df):
    add_id(market_df, MARKET_ID)
    add_id(news_df, NEWS_ID)


# In[ ]:


add_ids(market_train_df, news_train_df)

# In[ ]:


market_train_df["id"].max()

# In[ ]:


news_train_df["news_id"].max()

# In[ ]:


market_train_df[:1]


# In[ ]:


def add_confidence(df):
    # TODO change confidence by return proportion
    df["confidence"] = df["returnsOpenNextMktres10"] >= 0


# In[ ]:


add_confidence(market_train_df)

# In[ ]:


market_train_df[:10]

# In[ ]:


market_train_df[:1]

# In[ ]:


market_train_df.shape

# In[ ]:


market_train_df.id.tail()

# # full fill missing values

# In[ ]:


market_train_df.isnull().sum(axis=0)

# returnsClosePrevMktres1 , returnsOpenPrevMktres1  returnsClosePrevMktres10  and returnsOpenPrevMktres10 aren't used here for this model, I ignore them

# In[ ]:


news_train_df.isnull().sum(axis=0)

# In[ ]:


# empty string check
for col, dtype in zip(news_train_df.columns, news_train_df.dtypes):
    if dtype == np.dtype('O'):
        n_empty = (news_train_df[col] == "").sum()
        print("empty value in {}: {}".format(col, n_empty))

# In[ ]:


news_train_df.headlineTag.value_counts()


# It seems headlineTag values are categorical values. Therefore, I convert the column into categorical values.

# In[ ]:


def fill_missing_value_news_df(news_df):
    news_df.headlineTag.replace("", "UNKNONWN", inplace=True)


# In[ ]:


fill_missing_value_news_df(news_df=news_train_df)


# In[ ]:


def to_category_news_df(news_df):
    news_df.headlineTag = news_df.headlineTag.astype('category')


# In[ ]:


to_category_news_df(news_train_df)

# In[ ]:


news_train_df.dtypes

# # Feature Extraction

# In[ ]:


from abc import ABCMeta, abstractmethod


# In[ ]:


class FeatureTransformer(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, df):
        pass

    @abstractmethod
    def release_raw_field(self, df):
        pass


# ## news

# In[ ]:


concatable_features = ["subjects", "audiences", "headline", "provider", "headlineTag"]

# ### flatten subjects

# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer

# In[ ]:


import re


# In[ ]:


# re.sub(r"[\{\}\']", "", news_train_df.subjects.cat.categories.values[1]).split(", ")


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


def generate_flatten_one_hot_encoder(cat_values):
    flat_cats = flatten_category_complex(cat_values)
    flat_cats = list(set(itertools.chain.from_iterable(flat_cats)))
    flat_cats = np.asarray(list(flat_cats)).reshape(-1, 1)
    print("flat category size: {}".format(len(flat_cats)))
    encoder = MultiLabelBinarizer(sparse_output=True)
    encoder.fit(flat_cats)
    return encoder


# In[ ]:


# subjects_encoder = generate_flatten_one_hot_encoder(news_train_df.subjects.cat.categories.values)


# In[ ]:


# subjects_encoder.classes_


# In[ ]:


def binary_encode_cat_complex(encoder, target):
    return encoder.transform(flatten_category_complex(target))


# In[ ]:


# flat_subject_features = binary_encode_cat_complex(subjects_encoder, news_train_df.subjects) 


# In[ ]:


BATCH_SIZE = 5000

# In[ ]:


# BATCH_ITER = news_train_df.subjects.size // BATCH_SIZE \
#             + int(bool(news_train_df.subjects.size % BATCH_SIZE))


# In[ ]:


# def subjects_batch_generator(seq):
#     i = 0
#     while True:
#         if i < BATCH_ITER - 1:
#             yield seq[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
#         elif i == BATCH_ITER:
#             yield seq[i * BATCH_ITER:]
#         else:
#             break
#         i += 1


# In[ ]:


# class FlatSubjects(FeatureTransformer):
#     def transform(self, df):
#         #seq = df.subjects.astype("str")
#         seq = df.subjects
#         self.release_raw_field(df)
#         return subjects_encoder.transform(seq)

#     def release_raw_field(self, df):
#         df.drop(["subjects"], axis=1, inplace=True) 
#         gc.collect()


# ## flatten audience

# In[ ]:


audience_encoder = generate_flatten_one_hot_encoder(news_train_df.audiences)


# In[ ]:


class FlatAudience(FeatureTransformer):
    def transform(self, df):
        seq = df.audiences.astype("str")
        self.release_raw_field(df)
        seq = seq.apply(parse_category_complex)
        return audience_encoder.transform(seq.tolist())

    def release_raw_field(self, df):
        df.drop(["audiences"], axis=1, inplace=True)
        gc.collect()


# ## encode provider

# In[ ]:


provider_encoder = MultiLabelBinarizer(sparse_output=True)

# In[ ]:


provider_encoder.fit([news_train_df.provider.cat.categories.values])

# In[ ]:


provider_encoder.classes_

# In[ ]:


provider_encoder.classes_.size


# In[ ]:


class ProviderBinaryEncode(FeatureTransformer):
    def transform(self, df):
        seq = df.provider.astype("str")
        self.release_raw_field(df)
        seq = seq.apply(parse_category_complex)
        return provider_encoder.transform(seq)

    def release_raw_field(self, df):
        df.drop(["provider"], axis=1, inplace=True)
        gc.collect()


# # flatten sentimentClass

# In[ ]:


# class BinaryEncode()


# ## encode headlineTag

# In[ ]:


headlineTag_encoder = MultiLabelBinarizer(sparse_output=True)

# In[ ]:


headlineTag_encoder.fit([news_train_df.headlineTag.cat.categories.values])

# In[ ]:


headlineTag_encoder.classes_

# In[ ]:


headlineTag_encoder.classes_.size


# In[ ]:


class HeadlineTagBinaryEncode(FeatureTransformer):
    def transform(self, df):
        seq = df.headlineTag.astype("str")
        self.release_raw_field(df)
        seq = seq.apply(parse_category_complex)
        return headlineTag_encoder.transform(seq)

    def release_raw_field(self, df):
        df.drop(["headlineTag"], axis=1, inplace=True)
        gc.collect()


#  # remove unnecc

# In[ ]:


def remove_unnecessary_columns(market_df, news_df):
    # market_df.drop(["returnsOpenNextMktres10", "universe"], axis=1, inplace=True)
    market_df.drop(["assetName"], axis=1, inplace=True)
    news_df.drop(['time', 'sourceId', 'sourceTimestamp', "assetName"], axis=1, inplace=True)


# In[ ]:


def remove_unnecessary_columns_train(market_df, news_df):
    market_df.drop(["returnsOpenNextMktres10", "universe"], axis=1, inplace=True)
    remove_unnecessary_columns(market_df, news_df)


# In[ ]:


remove_unnecessary_columns_train(market_train_df, news_train_df)

# In[ ]:


gc.collect()

# In[ ]:


gc.collect()

# In[ ]:


import sys

# In[ ]:


for memory_info in ["{}: {}".format(v, sys.getsizeof(eval(v)) / 1000000000) for v in dir()]:
    print(memory_info)

# # link data and news

# In[ ]:


from scipy import sparse

# ## check assecName links

# In[ ]:


# headline = news_train_df.headline
# news_train_df.drop(["headline"], axis=1, inplace=True)

# subjects = news_train_df.subjects
# news_train_df.drop(["subjects"], axis=1, inplace=True)

# audiences = news_train_df.audiences
# news_train_df.drop(["audiences"], axis=1, inplace=True)

# categorical_column_values = [news_train_df[feature] for feature in categorical_features]

# news_train_df.drop(categorical_features, axis=1, inplace=True)


# In[ ]:


MAX_DAY_DIFF = 3
MULTIPLE_CODES_PATTERN = re.compile(r"[{}'']")
import itertools


class MarketNewsLinker(object):

    def __init__(self):
        self.market_df = None
        self.news_df = None
        self.link_df = None
        self.market_columns = None
        self.datatypes_before_aggregation = None

    def link_market_assetCode_and_news_assetCodes(self):
        assetCodes_in_markests = self.market_df.assetCode.unique()
        print("assetCodes pattern in markets: {}".format(len(assetCodes_in_markests)))
        assetCodes_in_news = self.news_df.assetCodes.unique()
        assetCodes_in_news_size = len(assetCodes_in_news)
        print("assetCodes pattern in news: {}".format(assetCodes_in_news_size))
        parse_multiple_codes = lambda codes: re.sub(r"[{}'']", "", str(codes)).split(", ")
        parsed_assetCodes_in_news = [parse_multiple_codes(str(codes)) for codes in assetCodes_in_news]
        # len(max(parsed_assetCodes_in_news, key=lambda x: len(x)))
        all_assetCode_type_in_news = list(set(itertools.chain.from_iterable(assetCodes_in_news)))
        # check linking
        links_assetCodes = [[[raw_codes, market_assetCode] for parsed_codes, raw_codes in
                             zip(parsed_assetCodes_in_news, assetCodes_in_news) if
                             str(market_assetCode) in parsed_codes] for market_assetCode in assetCodes_in_markests]
        links_assetCodes = list(itertools.chain.from_iterable(links_assetCodes))
        print("links for assetCodes: {}".format(len(links_assetCodes)))
        links_assetCodes = pd.DataFrame(links_assetCodes, columns=["newsAssetCodes", "marketAssetCode"],
                                        dtype='category')
        self.market_df = self.market_df.merge(links_assetCodes, left_on="assetCode", right_on="marketAssetCode")
        ## merge assetCodes links
        self.market_df.drop(["assetCode", "marketAssetCode"], axis=1, inplace=True)
        self.market_columns.remove("assetCode")

    def append_working_date_on_market(self):
        self.market_df["date"] = self.market_df.time.dt.date
        self.news_df["firstCreatedDate"] = self.news_df.firstCreated.dt.date
        self.news_df.firstCreatedDate = self.news_df.firstCreatedDate.astype(np.datetime64)

        working_dates = self.news_df.firstCreatedDate.unique()
        working_dates.sort()
        market_dates = self.market_df.date.unique().astype(np.datetime64)
        market_dates.sort()

        def find_prev_date(date):
            for diff_day in range(1, MAX_DAY_DIFF + 1):
                prev_date = date - np.timedelta64(diff_day, 'D')
                if len(np.searchsorted(working_dates, prev_date)) > 0:
                    return prev_date
            return None

        prev_news_days_for_market_day = np.apply_along_axis(arr=market_dates, func1d=find_prev_date, axis=0)

        date_df = pd.DataFrame(columns=["date", "prevDate"])
        date_df.date = market_dates

        date_df.prevDate = prev_news_days_for_market_day

        self.market_df.date = self.market_df.date.astype(np.datetime64)
        self.market_df = self.market_df.merge(date_df, left_on="date", right_on="date")

    def link_market_id_and_news_id(self):
        print("linking ids...")
        ## merge market and news
        link_df = self.market_df.merge(self.news_df, left_on=["newsAssetCodes", "date"],
                                       right_on=["assetCodes", "firstCreatedDate"])
        # remove news after market obs
        link_df = link_df[link_df["time"] > link_df["firstCreated"]]
        # self.link_df.sort_values(by=["time"],inplace=True)
        # self.link_df.drop_duplicates(subset=["id"], keep="last", inplace=True)

        prev_day_link_df = self.market_df.merge(
            self.news_df, left_on=["newsAssetCodes", "prevDate"],
            right_on=["assetCodes", "firstCreatedDate"])
        # self.market_df_prev_day_news.sort_values(by=["firstCreated"], inplace=True)
        del self.news_df
        del self.market_df
        gc.collect()

        print("updating market df....")
        self.market_df = pd.concat([link_df, prev_day_link_df])

        dropped_columns = ["time", "date", "prevDate", "newsAssetCodes",
                           "assetCodes",
                           "firstCreated", "firstCreatedDate", "news_id"]
        self.market_df.drop(dropped_columns, axis=1, inplace=True)
        for col in dropped_columns:
            if col in self.market_columns:
                self.market_columns.remove(col)

    def aggregate_day_asset_news(self):
        print("aggregating....")
        agg_func_map = {column: "mean" for column in self.market_df.columns.tolist()
                        if column not in self.market_columns
                        and column not in concatable_features}
        agg_func_map.update({col: "first"
                             for col in self.market_columns})
        agg_func_map.update({
            column: lambda x: ", ".join(x) if x.dtype == "object" else ", ".join([str(v) for v in x])
            for column in concatable_features
        })
        agg_func_map["headline"] = lambda x: " ".join(x)
        #         agg_func_map["news_id"] = lambda x: x
        print(agg_func_map)
        print(self.market_df.dtypes)
        gc.collect()
        self.market_df = self.market_df.groupby("id").agg(agg_func_map)
        self._update()
        self.fit_datatype()

    def _update(self):
        self.market_columns = self.market_df.columns.tolist()

    def fit_datatype(self):
        for col in self.market_df.columns:
            previous_dtype = self.datatypes_before_aggregation[col]
            if previous_dtype == np.dtype('float16') or previous_dtype == np.dtype(
                    'int16') or previous_dtype == np.dtype('int8') or previous_dtype == np.dtype("bool"):
                self.market_df[col] = self.market_df[col].astype("float16")
            elif previous_dtype == np.dtype("float64") or previous_dtype == np.dtype(
                    "float32") or previous_dtype == np.dtype('int32'):
                self.market_df[col] = self.market_df[col].astype("float32")

    def link(self, market_df, news_df):
        self.market_df = market_df
        self.news_df = news_df
        self.link_df = None

        self.market_columns = self.market_df.columns.tolist()
        self.datatypes_before_aggregation = {col: t for col, t in zip(self.market_columns, self.market_df.dtypes)}
        self.datatypes_before_aggregation.update(
            {col: t for col, t in zip(self.news_df.columns, self.news_df.dtypes)}
        )
        self.link_market_assetCode_and_news_assetCodes()

        self.append_working_date_on_market()

        self.link_market_id_and_news_id()
        self.aggregate_day_asset_news()

        print("linking done")
        return self.market_df

    def clear(self):
        del self.market_df
        self.market_df = None
        self.news_df = None
        self.link_df = None
        self.market_columns = None
        self.datatypes_before_aggregation = None


# In[ ]:


gc.collect()
linker = MarketNewsLinker()

# In[ ]:


get_ipython().run_cell_magic('time', '', 'market_train_df = linker.link(market_train_df, news_train_df)')

# In[ ]:


linker.market_df.dtypes

# In[ ]:


linker.market_df.columns

# In[ ]:


market_train_df[:10]

# In[ ]:


linker.clear()

# In[ ]:


del news_train_df

# In[ ]:


gc.collect()

# In[ ]:


gc.collect()

# # feature extraction II and dimension reduction

# In[ ]:


market_train_df.shape

# In[ ]:


from collections import OrderedDict

# In[ ]:


news_features = OrderedDict()


# In[ ]:


class UnionFeaturePipeline(object):

    def __init__(self, *args):
        if args == None:
            self.transformers = []
        else:
            self.transformers = list(args)

    def transform(self, df, include_sparse=True):
        if include_sparse:
            return sparse.hstack([transformer.transform(df) for transformer in self.transformers])
        return np.hstack([transformer.transform(df) for transformer in self.transformers])

    def add(self, transformer):
        self.transformers.append(transformer)


# In[ ]:


pipeline = UnionFeaturePipeline()

# ## audience

# In[ ]:


audience_transformer = FlatAudience()

# In[ ]:


market_train_df.columns

# In[ ]:


flat_audience_feature = audience_transformer.transform(market_train_df)

# In[ ]:


market_train_df.columns

# In[ ]:


news_features["audiences"] = flat_audience_feature

# In[ ]:


pipeline.add(audience_transformer)

# In[ ]:


print("audiences feature extraction has done.")

# In[ ]:


sys.getsizeof(pipeline)

# # provider

# In[ ]:


provier_transformer = ProviderBinaryEncode()

# In[ ]:


provider_binary = provier_transformer.transform(market_train_df)

# In[ ]:


news_features["provider"] = provider_binary

# In[ ]:


gc.collect()

# In[ ]:


market_train_df.dtypes

# In[ ]:


pipeline.add(provier_transformer)

# In[ ]:


print("provider feature extraction has done.")

# # headlineTag

# In[ ]:


headlineTag_transformer = HeadlineTagBinaryEncode()

# In[ ]:


headlineTag_category = headlineTag_transformer.transform(market_train_df)

# In[ ]:


news_features["headlineTag"] = headlineTag_category

# In[ ]:


market_train_df.shape

# In[ ]:


pipeline.add(headlineTag_transformer)

# In[ ]:


print("headlineTag feature extraction has done.")

# # headline

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

# In[ ]:


headline_vectorizer = CountVectorizer(decode_error="ignore",
                                      stop_words="english",
                                      strip_accents="unicode",
                                      max_features=3000,
                                      binary=True,
                                      dtype='int8')

# In[ ]:


binary_bow = headline_vectorizer.fit_transform(market_train_df.headline)

# In[ ]:


binary_bow.shape

# In[ ]:


headline_vectorizer.get_feature_names()

# In[ ]:


binary_bow[:10].todense()

# In[ ]:


market_train_df.drop(["headline"], axis=1, inplace=True)

# In[ ]:


gc.collect()


# In[ ]:


class HeadlineBinaryBow(FeatureTransformer):
    def transform(self, df):
        seq = df.headline
        self.release_raw_field(df)
        return headline_vectorizer.transform(seq)

    def release_raw_field(self, df):
        df.drop(["headline"], axis=1, inplace=True)
        gc.collect()


# In[ ]:


print("headline feature extraction has done.")

# In[ ]:


pipeline.add(HeadlineBinaryBow())

# In[ ]:


news_features["headline"] = binary_bow

# ## subjects

# In[ ]:


subjects_encoder = CountVectorizer(decode_error="ignore",
                                   strip_accents="unicode",
                                   max_features=5000,
                                   binary=True,
                                   dtype='int8')

# In[ ]:


# subjects_transformer = FlatSubjects()


# In[ ]:


flat_subjects_category = subjects_encoder.fit_transform(market_train_df.subjects)

# In[ ]:


news_features["subjects"] = flat_subjects_category


# In[ ]:


class SubjectsBinary(FeatureTransformer):
    def transform(self, df):
        seq = df.subjects
        self.release_raw_field(df)
        return subjects_encoder.transform(seq)

    def release_raw_field(self, df):
        df.drop(["subjects"], axis=1, inplace=True)
        gc.collect()


# In[ ]:


pipeline.add(SubjectsBinary())

# In[ ]:


market_train_df.drop(["subjects"], axis=1, inplace=True)

# In[ ]:


print("subjects feature extraction has done")

# # convert into trainable form

# In[ ]:


import lightgbm as lgb


# In[ ]:


# def to_Y(train_df):
#     return np.asarray(train_df.confidence)

# train_Y = to_Y(train_df=market_train_df)

# market_train_df.drop(["confidence"], axis=1, inplace=True)


# In[ ]:


def to_X(df, news_features, news_feature_names, additional_feature, additional_feature_names):
    market_obs_ids = df.id
    #     news_obs_ids = df.news_id
    news_obs_ids = []
    #    market_obs_times = df.time
    market_obs_times = []
    df.drop(["id"], axis=1, inplace=True)

    feature_names = df.columns.tolist()

    print(df.dtypes)
    if len(news_feature_names) > 0 and isinstance(news_features, np.ndarray):
        feature_names.extend(news_feature_names)
        row_indices = [market_id - 1 for market_id in news_obs_ids.tolist()]
        news_features = news_features[row_indices]
        X = np.hstack([X, news_features])
        del news_features

    if len(additional_feature_names) > 0 and (
            isinstance(additional_feature, np.ndarray) or sparse.issparse(additional_feature)):
        feature_names.extend(additional_feature_names)
        if sparse.issparse(additional_feature):
            X = sparse.hstack([df.values, additional_feature]).tocsr()
        else:
            X = np.hstack([df.values, additional_feature])
    del df
    gc.collect()
    return X, market_obs_ids, news_obs_ids, market_obs_times, feature_names


# In[ ]:


def to_train_dataset(df, train_X2, news_feature_names, train_size=0.8):
    def to_Y(df):
        return np.asarray(df.confidence)

    train_Y = to_Y(df=df)
    df.drop(["confidence"], axis=1, inplace=True)
    market_obs_ids = df.id
    #     news_obs_ids = df.news_id
    #    market_obs_times = df.time
    df.drop(["id"], axis=1, inplace=True)

    feature_names = df.columns.tolist()
    feature_names.extend(news_feature_names)

    train_X = df.values
    del df
    gc.collect()
    train_size = int(train_X.shape[0] * train_size)

    orders = np.argsort(market_obs_ids)

    train_X = train_X[orders]
    train_X2 = train_X2[orders.tolist()]
    train_Y = train_Y[orders]

    valid_X, valid_X2, valid_Y = train_X[train_size:], train_X2[train_size:], train_Y[train_size:]
    train_X, train_X2, train_Y = train_X[:train_size], train_X2[:train_size], train_Y[:train_size]

    train_X = lgb.Dataset(sparse.hstack([train_X, train_X2]), label=train_Y, feature_name=feature_names,
                          free_raw_data=False)
    del train_X2
    del train_Y
    gc.collect()

    valid_X = train_X.create_valid(sparse.hstack([valid_X, valid_X2]), label=valid_Y)
    del valid_X2
    del valid_Y

    return train_X, valid_X, market_obs_ids, feature_names


# In[ ]:


news_feature_names = [["{}_{}".format(name, i) for i in range(feature.shape[1])] for name, feature
                      in news_features.items()]

# In[ ]:


news_feature_names = list(itertools.chain.from_iterable(news_feature_names))

# In[ ]:


news_feature_names[-10:]

# In[ ]:


# news_features = np.hstack(news_features)
news_features = sparse.hstack(list(news_features.values()), dtype="int8").tocsr()

# In[ ]:


news_features.shape

# In[ ]:


gc.collect()

# In[ ]:


# %%time
# X, market_train_obs_ids, news_train_obs_ids, market_train_obs_times, feature_names = to_X(
#     market_train_df, None, [], news_features, news_feature_names
# )


# In[ ]:


X, valid_X, market_obs_ids, feature_names = to_train_dataset(market_train_df, news_features, news_feature_names,
                                                             train_size=0.8)

# In[ ]:


del news_features
del market_train_df
gc.collect()

# In[ ]:


len(feature_names)

# # create validation data

# # train model

# In[ ]:


# train_size = X.shape[0] // 5 * 4

# train_size

# valid_X, valid_Y = X[train_size:], train_Y[train_size:]

# X, train_Y = X[:train_size], train_Y[:train_size]

# X.shape

# valid_X.shape


# In[ ]:


feature_names

# In[ ]:


# X = lgb.Dataset(X, label=train_Y, feature_name=feature_names, free_raw_data=False)

# valid_X = X.create_valid(valid_X, label=valid_Y)


# In[ ]:


gc.collect()

# ## train

# In[ ]:


RANDOM_SEED = 10

# In[ ]:


hyper_params = {"objective": "binary", "boosting": "gbdt", "num_iterations": 500,
                "learning_rate": 0.02, "num_leaves": 31, "num_threads": 2,
                "seed": RANDOM_SEED, "early_stopping_round": 10
                }

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = lgb.train(params=hyper_params, train_set=X, valid_sets=[valid_X])')

# In[ ]:


for feature, imp in zip(model.feature_name(), model.feature_importance()):
    print("{}: {}".format(feature, imp))

# In[ ]:


del X

# In[ ]:


del valid_X

# In[ ]:


gc.collect()

# In[ ]:


# import seaborn as sns

# sns.set()

# sns.set_context("notebook")

# import matplotlib.pyplot as plt

# %matplotlib inline

# sns.barplot(x=model.feature_name(), y=model.feature_importance(), 
#             ax=plt.subplots(figsize=(20, 10))[1])


# ## `get_prediction_days` function
# 
# Generator which loops through each "prediction day" (trading day) and provides all market and news observations which occurred since the last data you've received.  Once you call **`predict`** to make your future predictions, you can continue on to the next prediction day.
# 
# Yields:
# * While there are more prediction day(s) and `predict` was called successfully since the last yield, yields a tuple of:
#     * `market_observations_df`: DataFrame with market observations for the next prediction day.
#     * `news_observations_df`: DataFrame with news observations for the next prediction day.
#     * `predictions_template_df`: DataFrame with `assetCode` and `confidenceValue` columns, prefilled with `confidenceValue = 0`, to be filled in and passed back to the `predict` function.
# * If `predict` has not been called since the last yield, yields `None`.

# ### **`predict`** function
# Stores your predictions for the current prediction day.  Expects the same format as you saw in `predictions_template_df` returned from `get_prediction_days`.
# 
# Args:
# * `predictions_df`: DataFrame which must have the following columns:
#     * `assetCode`: The market asset.
#     * `confidenceValue`: Your confidence whether the asset will increase or decrease in 10 trading days.  All values must be in the range `[-1.0, 1.0]`.
# 
# The `predictions_df` you send **must** contain the exact set of rows which were given to you in the `predictions_template_df` returned from `get_prediction_days`.  The `predict` function does not validate this, but if you are missing any `assetCode`s or add any extraneous `assetCode`s, then your submission will fail.

# Let's make random predictions for the first day:

# In[ ]:


from pandas.api.types import CategoricalDtype


# In[ ]:


def to_category_type(df, category_columns, categories_list):
    for col, categories in zip(category_columns, categories_list):
        cat_type = CategoricalDtype(categories=categories)
        df[col] = df[col].astype(cat_type)


# In[ ]:


def make_predictions(market_obs_df, news_obs_df, predictions_df):
    add_ids(market_obs_df, news_obs_df)
    fill_missing_value_news_df(news_obs_df)
    compress_dtypes(market_obs_df)
    compress_dtypes(news_obs_df)
    #     to_category_type(news_obs_df, category_columns=categorical_features,
    #                      categories_list= news_categories)
    #     encode_categorical_fields(news_df=news_obs_df)
    remove_unnecessary_columns(market_obs_df, news_obs_df)
    market_obs_df = linker.link(market_obs_df, news_obs_df)
    news_features = pipeline.transform(market_obs_df)
    X, market_train_obs_ids, news_train_obs_ids, market_train_obs_times, feature_names = to_X(market_obs_df,
                                                                                              None, [],
                                                                                              news_features,
                                                                                              news_feature_names)
    predictions_df.confidenceValue[[market_id - 1 for market_id in market_train_obs_ids]] = model.predict(X) * 2 - 1


# In[ ]:


days = env.get_prediction_days()


# In[ ]:


## random prediction for debug
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)

# ## Main Loop
# Let's loop through all the days and make our random predictions.  The `days` generator (returned from `get_prediction_days`) will simply stop returning values once you've reached the end.

# In[ ]:


from tqdm import tqdm

# In[ ]:


get_ipython().run_line_magic('time', '')
for (market_obs_df, news_obs_df, predictions_template_df) in tqdm(days):
    make_predictions(market_obs_df, news_obs_df, predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')

#  ## **`write_submission_file`** function
# 
# Writes your predictions to a CSV file (`submission.csv`) in the current working directory.

# In[ ]:


env.write_submission_file()

# In[ ]:


# We've got a submission file!
import os

print([filename for filename in os.listdir('.') if '.csv' in filename])

# As indicated by the helper message, calling `write_submission_file` on its own does **not** make a submission to the competition.  It merely tells the module to write the `submission.csv` file as part of the Kernel's output.  To make a submission to the competition, you'll have to **Commit** your Kernel and find the generated `submission.csv` file in that Kernel Version's Output tab (note this is _outside_ of the Kernel Editor), then click "Submit to Competition".  When we re-run your Kernel during Stage Two, we will run the Kernel Version (generated when you hit "Commit") linked to your chosen Submission.

# ## Restart the Kernel to run your code again
# In order to combat cheating, you are only allowed to call `make_env` or iterate through `get_prediction_days` once per Kernel run.  However, while you're iterating on your model it's reasonable to try something out, change the model a bit, and try it again.  Unfortunately, if you try to simply re-run the code, or even refresh the browser page, you'll still be running on the same Kernel execution session you had been running before, and the `twosigmanews` module will still throw errors.  To get around this, you need to explicitly restart your Kernel execution session, which you can do by pressing the Restart button in the Kernel Editor's bottom Console tab:
# ![Restart button](https://i.imgur.com/hudu8jF.png)
