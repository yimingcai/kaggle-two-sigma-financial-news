import gc
import itertools
import logging
import os
import re
import sys
from abc import abstractmethod, ABCMeta, ABC
from multiprocessing.pool import Pool
from pathlib import Path
from time import perf_counter
from typing import Union

import keras
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import torch
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

NEXT_MKTRES_10 = "returnsOpenNextMktres10"

CATEGORY_START_END_PATTERN = r"[\{\}\']"

SPLIT_PATTERN = r"[{}']"

logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.addHandler(logging.FileHandler("main.log"))

try:
    TEST_MARKET_DATA = Path(__file__).parent.joinpath("data/test/marketdata_sample.csv")
    TEST_NEWS_DATA = Path(__file__).parent.joinpath("data/test/news_sample.csv")
except NameError as e:
    TEST_MARKET_DATA = "data/test/marketdata_sample.csv"
    TEST_NEWS_DATA = "data/test/news_sample.csv"

# MODEL_TYPE = "mlp"
# MODEL_TYPE = "lgb"
MODEL_TYPE = "sparse_mlp"

MARKET_ID = "market_id"
NEWS_ID = "news_id"

np.random.seed(10)


class FeatureSetting(object):
    # remove_news_features = ["headline", "subjects", "headlineTag", "provider"]
    remove_news_features = []

    should_use_news_feature = True
    remove_raw_for_lag = True
    scale = True
    scale_type = "standard"
    # max_shift_date = 14
    max_shift_date = 10
    # since = date(2010, 1, 1)
    since = None
    should_use_prev_news = False


def main():
    logger.info("This model type is {}".format(MODEL_TYPE))
    # You can only call make_env() once, so don't lose it!
    env, market_train_df, news_train_df = load_train_dfs()

    # empty string check
    # for col, dtype in zip(news_train_df.columns, news_train_df.dtypes):
    #     if dtype == np.dtype('O'):
    #         n_empty = (news_train_df[col] == "").sum()
    #         logger.info("empty value in {}: {}".format(col, n_empty))

    market_preprocess = MarketPreprocess()
    market_train_df = market_preprocess.fit_transform(market_train_df)
    news_preprocess = NewsPreprocess()
    news_train_df = news_preprocess.fit_transform(news_train_df)

    features = Features()
    market_train_df, news_train_df = features.fit_transform(market_train_df, news_train_df)
    logger.info("First feature extraction has done")

    max_day_diff = 3
    gc.collect()

    # In[ ]:
    if FeatureSetting.should_use_news_feature:
        linker = MarketNewsLinker(max_day_diff)
        linker.link(market_train_df, news_train_df)
        del news_train_df
        del market_train_df
        gc.collect()
        market_train_df = linker.create_new_market_df()
        linker.clear()
        gc.collect()
    else:
        linker = None

        # In[ ]:
        #
        # from collections import OrderedDict

        # # feature extraction II and dimension reduction

        # In[ ]:
    # feature_matrix = features.get_linked_feature_matrix(market_train_df)
    # features.clear()
    # gc.collect()

    model = ModelWrapper.generate(MODEL_TYPE)

    # In[ ]:
    # logger.info("dtypes before train:")
    # logger.info(market_train_df.dtypes)
    # for col in market_train_df.columns:
    #     logger.info("{} has nan: {}".format(col, market_train_df[col].isnull().any()))
    # print(market_train_df["returnsClosePrevMktres10_lag_7_max"])

    market_train_df, _ = model.create_dataset(market_train_df, features, train_batch_size=1024,
                                              valid_batch_size=1024)
    gc.collect()
    model.train(sparse_input=True)
    model.clear()

    days = env.get_prediction_days()

    predictor = Predictor(linker, model, features, market_preprocess, news_preprocess)
    predictor.predict_all(days, env)
    # In[ ]:

    # In[ ]:

    logger.info('Done!')

    # In[ ]:

    env.write_submission_file()

    logger.info([filename for filename in os.listdir('.') if '.csv' in filename])


def measure_time(func):
    def inner(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start
        logger.info("%s took %.6f sec", func.__name__, duration)
        return result

    return inner


class UnionFeaturePipeline(object):

    def __init__(self, *args):
        if args is None:
            self.transformers = []
        else:
            self.transformers = list(args)

    def transform(self, df, include_sparse=True):
        feature_columns = []
        for transformer in self.transformers:
            if isinstance(transformer, NullTransformer):
                transformer.transform(df)
            elif isinstance(transformer, DfTransformer):
                df = transformer.transform(df)
            else:
                feature_columns.append(transformer.transform(df))

        if include_sparse:
            return df, sparse.hstack(feature_columns, format="csr")
        if len(feature_columns) == 0:
            return df, None
        return df, np.hstack(feature_columns)

    def add(self, transformer):
        self.transformers.append(transformer)


# In[ ]:
# TODO change make_column_transformer
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0


class ModelWrapper(ABC):

    def __init__(self, **kwargs):
        self.model = None
        super().__init__(**kwargs)

    @abstractmethod
    def predict(self, X: np.ndarray):
        return None

    @abstractmethod
    def train(self, **kwargs):
        return self

    @staticmethod
    def generate(model_type):
        if model_type == "lgb":
            return LgbWrapper()
        elif model_type == "mlp":
            return MLPWrapper()
        elif model_type == "sparse_mlp":
            return SparseMLPWrapper()
        else:
            raise ValueError("unknown model type: {}".format(model_type))

    @staticmethod
    def split_train_validation(train_X, train_Y, train_size):
        # train_size = int(train_X.shape[0] * train_size)
        train_size = int(len(train_X) * train_size)
        # orders = np.argsort(market_obs_ids).tolist()
        # train_X = train_X[orders]
        # train_Y = train_Y[orders]
        valid_X, valid_Y = train_X[train_size:], train_Y[train_size:]
        train_X, train_Y = train_X[:train_size], train_Y[:train_size]

        # if train_X2 is not None and train_X2.shape[0] > 0:
        #     train_X2 = train_X2[orders]
        #     valid_X2 = train_X2[train_size:]
        #     train_X2 = train_X2[:train_size]
        # else:
        #     valid_X2 = None

        return train_X, valid_X, train_Y, valid_Y

    @staticmethod
    def to_x_y(df):
        def to_Y(df):
            return np.asarray(df.confidence)

        train_Y = to_Y(df=df)
        df.drop(["confidence"], axis=1, inplace=True)
        market_obs_ids = df[MARKET_ID]
        #     news_obs_ids = df[NEWS_ID]
        #    market_obs_times = df.time
        df.drop([MARKET_ID], axis=1, inplace=True)
        # feature_names = df.columns.tolist()
        # if is_not_empty(news_feature_names):
        #     feature_names.extend(news_feature_names)
        # train_X = df.values
        # del df
        # gc.collect()
        return train_Y, market_obs_ids

    @abstractmethod
    def create_dataset(self, market_train, features, train_batch_size, valid_batch_size):
        return None, None


class LgbWrapper(ModelWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @measure_time
    def train(self, **kwargs):
        # len(self.feature_names)
        gc.collect()
        # In[ ]:
        RANDOM_SEED = 10
        # In[ ]:

        hyper_params = {"objective": "binary", "boosting": "gbdt", "num_iterations": 500,
                        "learning_rate": 0.2, "num_leaves": 2500,
                        "num_threads": 2, "max_bin": 205, 'min_data_in_leaf': 210,
                        "seed": RANDOM_SEED, "early_stopping_round": 10
                        }
        # ## train
        # In[ ]:
        model = lgb.train(params=hyper_params, train_set=self.x, valid_sets=[self.valid_X])
        # In[ ]:
        for feature, imp in zip(model.feature_name(), model.feature_importance()):
            logger.info("{}: {}".format(feature, imp))
        # In[ ]:
        #
        del self.x
        # In[ ]:
        del self.valid_X
        # In[ ]:
        gc.collect()
        # In[ ]:
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def create_dataset(self, df, features, train_batch_size, valid_batch_size):
        y, self.market_obs_ids = ModelWrapper.to_x_y(df)
        train_size = 0.8
        self.x, self.valid_X, y, valid_Y = ModelWrapper.split_train_validation(
            features, y,
            train_size
        )
        # if is_not_empty(train_X2):
        #     self.x = sparse.hstack([self.x, train_X2])
        self.x = lgb.Dataset(self.x, label=y,
                             free_raw_data=False)

        self.valid_X = self.x.create_valid(self.valid_X, label=valid_Y)
        del valid_Y
        return None, None


def is_not_empty(list_like):
    if list_like is None:
        return False
    if isinstance(list_like, np.ndarray) or sparse.issparse(list_like):
        return list_like.shape[0] > 0
    return len(list_like) > 0


class MarketNewsLinker(object):

    def __init__(self, max_day_diff):
        self.market_df = None
        self.news_df = None
        self.market_columns = None
        self.max_day_diff = max_day_diff
        self.datatypes_before_aggregation = None
        # self.concatable_features = concatable_fields
        self.news_columns = None

    def link_market_assetCode_and_news_assetCodes(self):
        assetCodes_in_markests = self.market_df.assetCode.unique().tolist()
        logger.info("assetCodes pattern in markets: {}".format(len(assetCodes_in_markests)))
        assetCodes_in_news = self.news_df.assetCodes.unique()
        assetCodes_in_news_size = len(assetCodes_in_news)
        logger.info("assetCodes pattern in news: {}".format(assetCodes_in_news_size))
        parse_multiple_codes = lambda codes: re.sub(SPLIT_PATTERN, "", str(codes)).split(", ")
        parsed_assetCodes_in_news = [parse_multiple_codes(str(codes)) for codes in assetCodes_in_news]
        # len(max(parsed_assetCodes_in_news, key=lambda x: len(x)))
        # all_assetCode_type_in_news = list(set(itertools.chain.from_iterable(assetCodes_in_news)))
        # check linking
        links_assetCodes = [[[raw_codes, market_assetCode] for parsed_codes, raw_codes in
                             zip(parsed_assetCodes_in_news, assetCodes_in_news) if
                             str(market_assetCode) in parsed_codes] for market_assetCode in assetCodes_in_markests]
        links_assetCodes = list(itertools.chain.from_iterable(links_assetCodes))
        logger.info("links for assetCodes: {}".format(len(links_assetCodes)))
        links_assetCodes = pd.DataFrame(links_assetCodes, columns=["newsAssetCodes", "marketAssetCode"],
                                        dtype='category')

        logger.info(links_assetCodes.shape)
        # self.market_df = self.market_df.merge(links_assetCodes, left_on="assetCode", right_on="marketAssetCode",
        #                                       copy=False, how="left", left_index=True)
        self.market_df = self.market_df.merge(links_assetCodes, left_on="assetCode", right_on="marketAssetCode",
                                              copy=False, how="left")
        logger.info(self.market_df.shape)
        # merge assetCodes links
        self.market_df.drop(["marketAssetCode"], axis=1, inplace=True)
        # self.market_df.drop(["marketAssetCode"], axis=1)

    def append_working_date_on_market(self):
        self.market_df["date"] = self.market_df.time.dt.date
        self.news_df["firstCreatedDate"] = self.news_df.firstCreated.dt.date
        self.news_df.firstCreatedDate = self.news_df.firstCreatedDate.astype(np.datetime64)

        working_dates = self.news_df.firstCreatedDate.unique()
        working_dates.sort()
        market_dates = self.market_df.date.unique().astype(np.datetime64)
        market_dates.sort()

        def find_prev_date(date):
            for diff_day in range(1, self.max_day_diff + 1):
                prev_date = date - np.timedelta64(diff_day, 'D')
                if len(np.searchsorted(working_dates, prev_date)) > 0:
                    return prev_date
            return None

        prev_news_days_for_market_day = np.apply_along_axis(arr=market_dates, func1d=find_prev_date, axis=0)

        date_df = pd.DataFrame(columns=["date", "prevDate"])
        date_df.date = market_dates

        date_df.prevDate = prev_news_days_for_market_day

        self.market_df.date = self.market_df.date.astype(np.datetime64)
        self.market_df = self.market_df.merge(date_df, left_on="date", right_on="date", how="left")

    def link_market_id_and_news_id(self):
        logger.info("linking ids...")
        self.news_columns = self.news_df.columns.tolist()
        # merge market and news
        market_link_columns = [MARKET_ID, "time", "newsAssetCodes", "date", "prevDate"]
        news_link_df = self.news_df[["assetCodes", "firstCreated", "firstCreatedDate", NEWS_ID]]
        self.news_df.drop(["assetCodes", "firstCreated", "firstCreatedDate"], axis=1, inplace=True)
        link_df = self.market_df[market_link_columns].merge(news_link_df, left_on=["newsAssetCodes", "date"],
                                                            right_on=["assetCodes", "firstCreatedDate"], how='left')
        # remove news after market obs
        link_df = link_df[link_df["time"] > link_df["firstCreated"]]
        link_df.drop(["time", "newsAssetCodes", "date", "prevDate"], axis=1, inplace=True)
        # link_df = link_df.drop(["time", "newsAssetCodes", "date", "prevDate"], axis=1)

        # self.link_df.sort_values(by=["time"],inplace=True)
        # self.link_df.drop_duplicates(subset=[MARKET_ID], keep="last", inplace=True)

        if FeatureSetting.should_use_prev_news:
            prev_day_link_df = self.market_df[market_link_columns].merge(
                news_link_df, left_on=["newsAssetCodes", "prevDate"],
                right_on=["assetCodes", "firstCreatedDate"])
            prev_day_link_df = prev_day_link_df[
                prev_day_link_df["time"] - pd.Timedelta(days=1) < prev_day_link_df["firstCreated"]]
            prev_day_link_df = prev_day_link_df.drop(
                ["time", "newsAssetCodes", "date", "prevDate"], axis=1, inplace=True)
            # prev_day_link_df = prev_day_link_df.drop(["time", "newsAssetCodes", "date", "prevDate"], axis=1)

        del news_link_df
        gc.collect()

        if FeatureSetting.should_use_prev_news:
            # link_df = pd.concat([link_df, prev_day_link_df])
            link_df = link_df.append(prev_day_link_df)
            del prev_day_link_df
            gc.collect()

        self.market_df = self.market_df.merge(link_df, on=MARKET_ID, how="left", copy=False)
        # self.market_df = self.market_df.merge(link_df, on=MARKET_ID, how="left")
        del link_df
        gc.collect()

        # self.market_df_prev_day_news.sort_values(by=["firstCreated"], inplace=True)
        # self.market_df = self.market_df.merge(self.news_df, on=NEWS_ID, how="left", copy=False, left_index=True).compute()
        # self.market_df = self.market_df.merge(self.news_df, on=NEWS_ID, how="left", copy=False)
        logger.info("shape after append news" + str(self.market_df.shape))

        #
        # del news_link_df
        # gc.collect()
        #
        # del prev_day_link_df
        # gc.collect()

    def aggregate_day_asset_news(self):
        logger.info("aggregating....")
        agg_func_map = {column: "mean" for column in self.market_df.columns.tolist()
                        if column == "marketCommentary" or column not in self.market_columns}
        agg_func_map.update({col: "first"
                             for col in self.market_columns})
        # agg_func_map.update({
        #     column: lambda x: ", ".join(x) if x.dtype == "object" else ", ".join([str(v) for v in x])
        #     for column in self.concatable_features
        # })
        agg_func_map[NEWS_ID] = lambda x: x.tolist()
        # agg_func_map["headline"] = lambda x: " ".join(x) if x is not None else ""
        #         agg_func_map[NEWS_ID] = lambda x: x
        logger.info(agg_func_map)
        logger.info(self.market_df.dtypes)
        # gc.collect()
        # id_date_groups = self.market_df.groupby(MARKET_ID)
        # with Pool(2) as pool:
        #     id_date_groups = pool.map(lambda group: group.agg(), id_date_groups)
        # self.market_df = pd.concat(id_date_groups)

        self.market_df = self.market_df.groupby(MARKET_ID).agg(agg_func_map)
        # self.market_df = [group[1] for group in self.market_df.groupby(MARKET_ID)]
        #
        # with Pool(self.pool) as pool:
        #     self.market_df = pool.map(lambda group: group.agg(agg_func_map),
        #                               self.market_df
        #                               )
        logger.info("the aggregation for each group has done")
        # self.market_df = pd.concat(self.market_df)
        self._update_inner_data()
        # self.fit_datatype()

    def _update_inner_data(self):
        self.market_columns = self.market_df.columns.tolist()

    #
    # def fit_datatype(self):
    #     for col in self.market_df.columns:
    #         previous_dtype = self.datatypes_before_aggregation[col]
    #         # print(col)
    #         # print(previous_dtype)
    #         # print(self.market_df[col].dtype)
    #         if previous_dtype == np.dtype('float16') or previous_dtype == np.dtype(
    #                 'int16') or previous_dtype == np.dtype('int8') or previous_dtype == np.dtype("bool"):
    #             self.market_df[col] = self.market_df[col].astype("float16")
    #         elif previous_dtype == np.dtype("float64") or previous_dtype == np.dtype(
    #                 "float32") or previous_dtype == np.dtype('int32'):
    #             self.market_df[col] = self.market_df[col].astype("float32")

    @measure_time
    def link(self, market_df, news_df, pool=4):
        self.market_df = market_df
        self.news_df = news_df
        self.pool = pool
        self.market_columns = self.market_df.columns.tolist()
        self.datatypes_before_aggregation = {col: t for col, t in zip(self.market_columns, self.market_df.dtypes)}
        self.datatypes_before_aggregation.update(
            {col: t for col, t in zip(self.news_df.columns, self.news_df.dtypes)}
        )
        self.link_market_assetCode_and_news_assetCodes()

        self.append_working_date_on_market()

        return self.link_market_id_and_news_id()

    @measure_time
    def create_new_market_df(self):
        logger.info("updating market df....")
        dropped_columns = ["date", "prevDate", "newsAssetCodes",
                           "assetCodes",
                           "firstCreated", "firstCreatedDate"]
        logger.info(self.market_df.columns)
        self.market_df.drop(dropped_columns, axis=1, inplace=True)
        self.market_df.sort_values(by=MARKET_ID, inplace=True)
        self.aggregate_day_asset_news()
        logger.info("linking done")
        return self.market_df

    #
    # def full_fill_new_columns(self, old_columns, new_columns):
    #     for col in set(new_columns) - set(old_columns):
    #         if "int" in str(self.market_df[col].dtype) or "float" in str(self.market_df[col].dtype):
    #             self.market_df[col] = self.market_df[col].fillna(0)
    #         elif col == NEWS_ID:
    #             self.market_df[col] = self.market_df[col].fillna([])
    #         elif self.market_df[col].dtype == np.dtype("object"):
    #             self.market_df[col] = self.market_df[col].fillna("")
    #         elif self.market_df[col].dtype.name == "category":
    #             if "" not in self.market_df[col].cat.categories:
    #                 self.market_df[col] = self.market_df[col].cat.add_categories("")
    #             self.market_df[col] = self.market_df[col].fillna("")
    #         elif self.market_df[col].dtype == np.dtype("bool"):
    #             self.market_df[col] = self.market_df[col].fillna(False)

    def clear(self):
        del self.market_df
        self.market_df = None
        self.news_df = None
        self.market_columns = None
        self.datatypes_before_aggregation = None


def compress_dtypes(news_df):
    for col, dtype in zip(news_df.columns, news_df.dtypes):
        if dtype == np.dtype('float64'):
            news_df[col] = news_df[col].astype("float32")
        if dtype == np.dtype('int64'):
            news_df[col] = news_df[col].astype("int32")


#
# def to_X(df, news_features, news_feature_names, additional_feature, additional_feature_names):
#     # sort_indices = df[MARKET_ID].values.argsort()
#     # df.sort_values(by=MARKET_ID, axis=0, inplace=True)
#     # market_obs_ids = df[MARKET_ID]
#     # print(market_obs_ids)
#
#     # if news_features is not None:
#     #     news_features = news_features[sort_indices]
#     # if additional_feature is not None:
#     #     additional_feature = additional_feature[sort_indices]
#     #     news_obs_ids = df[NEWS_ID]
#     # news_obs_ids = []
#     #    market_obs_times = df.time
#     # market_obs_times = []
#     df.drop([MARKET_ID], axis=1, inplace=True)
#
#     feature_names = df.columns.tolist()
#
#     # logger.info(df.dtypes)
#     #
#     # if len(news_feature_names) > 0 and isinstance(news_features, np.ndarray):
#     #     feature_names.extend(news_feature_names)
#     #     row_indices = [market_id for market_id in market_obs_ids]
#     #     news_features = news_features[row_indices]
#     #     X = np.hstack([df.values, news_features])
#     #     del news_features
#     #
#     # elif len(additional_feature_names) > 0 and (
#     #         isinstance(additional_feature, np.ndarray) or sparse.issparse(additional_feature)):
#     #     feature_names.extend(additional_feature_names)
#     #     if sparse.issparse(additional_feature):
#     #         X = sparse.hstack([df.values, additional_feature]).tocsr()
#     #     else:
#     #         X = np.hstack([df.values, additional_feature])
#     # else:
#     #     X = df.values
#
#     return X, market_obs_ids, news_obs_ids, market_obs_times, feature_names


def load_train_dfs():
    try:
        from kaggle.competitions import twosigmanews
        env = twosigmanews.make_env()
        (market_train_df, news_train_df) = env.get_training_data()
    except:
        market_train_df = pd.read_csv(TEST_MARKET_DATA, encoding="utf-8", engine="python")
        news_train_df = pd.read_csv(TEST_NEWS_DATA, encoding="utf-8", engine="python")
        env = None
    return env, market_train_df, news_train_df


class TorchDataset(Dataset):
    def __init__(self, matrix, labels, transformers=None):
        self._matrix = matrix
        self._labels = labels
        self._transformers = transformers
        self.n_features = matrix.shape[-1]

    def __getitem__(self, index):
        item = self._matrix[index, :]
        if self._transformers is None:
            return item, torch.Tensor(self._labels[index:index + 1])
        return self._transformers(item), torch.Tensor(self._labels[index:index + 1])

    def __len__(self):
        return self._matrix.shape[0]


class TorchDataLoader(DataLoader):

    def __init__(self, dataset: TorchDataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn)

    def __len__(self):
        return len(self.dataset)


def create_data_loader(matrix: Union[np.ndarray, sparse.coo_matrix, sparse.csr_matrix],
                       labels: np.ndarray, batch_size: int, shuffle: bool):
    if np.isnan(labels).any():
        raise ValueError("remove nan from labels")
    if isinstance(matrix, np.ndarray):
        if np.isnan(matrix).any():
            raise ValueError("remove nan from feature matrix")

    # elif sparse.issparse(matrix):
    #     if len(sparse.find(np.nan)[1]) > 0:
    #         raise ValueError("remove nan from feature matrix")

    def transformers(item):
        item = item.astype("float32")
        if sparse.issparse(matrix):
            # item = torch.sparse.FloatTensor(
            #     torch.LongTensor(item.nonzero()),
            #     torch.FloatTensor(item.data),
            #     torch.Size(item.shape))
            return torch.from_numpy(item.todense().A1)
        item = torch.from_numpy(item)
        return item

    dataset = TorchDataset(matrix, labels.astype("uint8").reshape((-1, 1)), transformers)

    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class BaseMLPClassifier(nn.Module):

    def __init__(self, fc_layer_params: list):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Linear(**params),
                nn.BatchNorm1d(params["out_features"]),
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            for i, params in enumerate(fc_layer_params[:-1])
        ]
        for layer in layers:
            layer.apply(self.init_weights)

        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(**fc_layer_params[-1])
        # if self.output_layer.out_features == 1:
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.fc_layers(x)
        out = self.output_layer(out)
        # if self.output_layer.out_features == 1:
        out = self.sigmoid(out)
        # out = self.softmax(out)
        return out

    # # self.softmax = nn.Softmax()


class BaseMLPTrainer(object):
    def __init__(self, model, loss_function, score_function, optimizer_factory):
        self.model: nn.Module = model
        # self.loss_function = nn.BCELoss()
        self.loss_function = loss_function
        self.score_function = score_function

        self.optimiser = optimizer_factory(self.model)

        self.train_data_loader = None
        self.valid_data_loader = None

        self.n_epoch = None
        self._current_epoch = 0
        self.train_losses = []
        self.train_scores = []
        self.valid_losses = []
        self.valid_scores = []

        self._current_max_valid_score = 0
        self._early_stop_count = 0

        self.save_name = "twosigma.model"

    def train(self, train_data_loader, valid_data_loader, n_epochs):
        self.clear_history()

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.n_epoch = n_epochs

        logger.info("train with: {}".format(self.train_data_loader.dataset._matrix.shape))
        logger.info("valid with: {}".format(self.valid_data_loader.dataset._matrix.shape))

        iterator = tqdm(range(n_epochs))
        for epoch in iterator:
            self._current_epoch = epoch + 1
            logger.info("training %d epoch / n_epochs", self._current_epoch)

            self._train_epoch()
            self._valid_epoch()

            if self.valid_scores[-1] <= self._current_max_valid_score:
                self._early_stop_count += 1
            else:
                logger.info("validation score is improved from %.3f to %.3f",
                            self._current_max_valid_score, self.valid_scores[-1])
                self._current_max_valid_score = self.valid_scores[-1]
                self._early_stop_count = 0
                self.save_models()

            if self._early_stop_count >= 10:
                logger.info("======early stopped=====")
                self.model.load_state_dict(torch.load(self.save_name))
                iterator.close()
                break

        logger.info("train done!")

    def clear_history(self):
        self.n_epoch = None
        self._current_epoch = 0

        self.train_losses = []
        self.train_scores = []
        self.valid_losses = []
        self.valid_scores = []

        self._current_max_valid_score = 0
        self._early_stop_count = 0

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for i, data in enumerate(self.train_data_loader):
            inputs, labels = data
            # print("batch data size {}".format(inputs.size()))

            self.optimiser.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)

            loss.backward()
            self.optimiser.step()
            total_loss += loss.item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self.train_data_loader)
        logger.info("******train loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))
        self.train_losses.append(avg_loss)

    def _valid_epoch(self):
        total_loss = 0.0

        all_labels = []
        all_outputs = []
        self.model.eval()
        for i, data in enumerate(self.valid_data_loader):
            inputs, labels = data
            outputs = self.model(inputs)
            all_labels.append(labels.detach().numpy())
            all_outputs.append(outputs.detach().numpy())
            loss = self.loss_function(outputs, labels)

            total_loss += loss.item()
            if i % 2000 == 1999:
                logger.info('[%d, %5d] validation loss: %.7f' %
                            (self._current_epoch, i + 1, total_loss / (i + 1)))

        avg_loss = total_loss / len(self.valid_data_loader)
        self.valid_losses.append(avg_loss)
        logger.info("******valid loss at epoch %d: %.7f :" % (self._current_epoch, avg_loss))

        all_outputs = np.vstack(all_outputs).reshape((-1))
        all_labels = np.vstack(all_labels).reshape((-1))
        score = self.score_function(all_outputs, all_labels)
        logger.info("******valid score at epoch %d: %.3f :" % (self._current_epoch, score))
        self.valid_scores.append(score)

    def save_models(self):
        torch.save(self.model.state_dict(), self.save_name)
        logger.info("Checkpoint saved")


class MLPWrapper(ModelWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, x: Union[np.ndarray, sparse.spmatrix]):
        logger.info("predicting %d samples...".format(x.shape[0]))
        self.model.eval()
        if sparse.issparse(x):
            x = x.todense()
        x = torch.from_numpy(x.astype("float32"))
        return self.model(x).detach().numpy().reshape((-1))

    def train(self, **kwargs):
        classes = 1
        model = BaseMLPClassifier(
            [{"in_features": self.train_data_loader.dataset.n_features, "out_features": 128, "bias": True},
             {"in_features": 128, "out_features": 64, "bias": True},
             {"in_features": 64, "out_features": 16, "bias": True},
             {"in_features": 16, "out_features": classes, "bias": True},
             ]
        )

        def score_function(predicted, labels):
            return roc_auc_score(labels, predicted)

        optimizer_factory = lambda model: optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
        trainer = BaseMLPTrainer(model, loss_function=nn.BCELoss(),
                                 score_function=score_function,
                                 optimizer_factory=optimizer_factory)

        trainer.train(self.train_data_loader, self.valid_data_loader, 50)

        self.model = model
        return self

    def create_dataset(self, market_train, features, train_batch_size, valid_batch_size):
        feature_names, market_obs_ids, market_train, labels = ModelWrapper.to_x_y(market_train)
        logger.info("concatenating train x....")
        market_train = market_train.astype("float32")
        if is_not_empty(features):
            features = features.astype("float32")
            market_train = sparse.hstack([market_train, features], format="csr")
        market_train, valid_matrix, labels, valid_labels, = ModelWrapper.split_train_validation(
            market_train,
            labels,
            train_size=0.8)
        logger.info("creating torch dataset....")
        market_train = market_train.astype("float32")
        valid_matrix = valid_matrix.astype("float32")
        self.train_data_loader = create_data_loader(market_train, labels, batch_size=train_batch_size, shuffle=True)
        self.valid_data_loader = create_data_loader(valid_matrix, valid_labels, batch_size=valid_batch_size,
                                                    shuffle=True)
        logger.info("torch dataset is created!")
        return None, None


class Preprocess(object):

    def __init__(self):
        self.transformers = []
        # self.transformers = [
        #     (col, LogTransformer(), col) for col in self.get_log_normal_columns()
        # ]
        # # self.transformers.extend([(col, TahnEstimators(), col) for col in self.get_columns_scaled()])
        # # self.transformers.extend([(col, ReshapeInto2d(), col) for col in self.get_columns_scaled()])
        # if FeatureSetting.scale_type.lower() == "minmax":
        #     self.transformers.extend([(col, MinMaxScaler(copy=False), col) for col in self.get_columns_scaled()])
        # elif FeatureSetting.scale_type.lower() == "standard":
        #     self.transformers.extend([(col, StandardScaler(copy=False), col) for col in self.get_columns_scaled()])
        # self.transformers.extend([(col, SimpleImputer(strategy="median"), col)
        #                           for col in self.get_fill_numeric_missing()])
        # self.transformers.extend([(col, SimpleImputer(strategy="constant", fill_value="UNKNOWN"), col)
        #                           for col in self.get_label_object_missing()])
        # self.transformers.extend([(col, SimpleImputer(strategy="constant", fill_value=""), col)
        #                           for col in self.get_sentence_missing()])

    def fit_transform(self, df: pd.DataFrame):
        for new_col_name, transformer, col_name in self.transformers:
            if not new_col_name:
                new_col_name = col_name
            df[new_col_name] = transformer.fit_transform(to_2d_array(df[col_name]))
        return df

    def transform(self, df: pd.DataFrame):
        for new_col_name, transformer, col_name in self.transformers:
            if not new_col_name:
                new_col_name = col_name
            df[new_col_name] = transformer.transform(to_2d_array(df[col_name]))
        return df


class MarketPreprocess(Preprocess):
    # COLUMNS_NORMALIZED = ['volume', 'close', 'open',
    #                       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
    #                       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
    #                       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    # LOG_NORMAL_FIELDS = ['volume', 'close', 'open',
    #                      'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                      'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
    #                      'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
    #                      'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    # LOG_NORMAL_FIELDS = []
    # COLUMNS_WITH_NUMERIC_MISSING = ['volume', 'close', 'open',
    #                                 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                                 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
    #                                 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
    #                                 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']

    def __init__(self):
        super().__init__()
        transformers = []

        if FeatureSetting.since is not None:
            transformers.append(DateFilterTransformer(FeatureSetting.since, "time"))

        transformers.extend([
            IdAppender(MARKET_ID),
            ConfidenceAppender(),
            LagAggregationTransformer(lags=[3, 5, 10], shift_size=1, scale=True, n_pool=3)
        ])

        self.pipeline: UnionFeaturePipeline = UnionFeaturePipeline(
            *transformers
        )

    def fit_transform(self, df: pd.DataFrame):
        df = super().fit_transform(df)
        return self.pipeline.transform(df, include_sparse=False)[0]

    def transform(self, df: pd.DataFrame):
        df = super().transform(df)
        return self.pipeline.transform(df, include_sparse=False)[0]


class NewsPreprocess(Preprocess):
    # COLUMNS_SCALED = [
    #     'takeSequence',
    #     'bodySize', 'companyCount',
    #     'sentenceCount', 'wordCount',
    #     # 'firstMentionSentence',
    #     # 'relevance',
    #     # 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
    #     'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
    #     'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
    #     'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
    #     'volumeCounts7D']
    # LOG_NORMAL_FIELDS = [
    #     'bodySize',
    #     'sentenceCount', 'wordCount',
    #     # 'firstMentionSentence',
    #     # 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
    #     'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
    #     'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
    #     'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
    #     'volumeCounts7D']

    # LABEL_OBJECT_FIELDS = [
    #     #     'sourceId', 'headlineTag'
    # #     # ]
    #
    # SENTENCE_FIELDS = ["headline"]

    def __init__(self):
        super().__init__()
        transformers = []

        if FeatureSetting.since is not None:
            transformers.append(DateFilterTransformer(FeatureSetting.since, "firstCreated"))
        transformers.append(IdAppender(NEWS_ID))
        self.pipeline: UnionFeaturePipeline = UnionFeaturePipeline(
            *transformers
        )

    def fit_transform(self, df: pd.DataFrame):
        df = super().fit_transform(df)
        return self.pipeline.transform(df, include_sparse=False)[0]

    def transform(self, df: pd.DataFrame):
        df = super().transform(df)
        return self.pipeline.transform(df, include_sparse=False)[0]


#
# def flatten_category_complex(cat_values):
#     flat_cats = [re.sub(CATEGORY_START_END_PATTERN, "", value).split(", ")
#                  for value in cat_values]
#     return flat_cats

class ReshapeInto2d(FunctionTransformer):
    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(to_2d_array, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)


def to_2d_array(x):
    array = x
    if isinstance(x, pd.Series):
        array = array.values
    if len(array.shape) == 1:
        array = array.reshape((-1, 1))
    return array


class LogTransformer(FunctionTransformer):

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(LogTransformer.to_log, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    @staticmethod
    def to_log(x):
        input_ = x
        # input_ = input_
        return np.log1p(input_)


class RavelTransformer(FunctionTransformer):

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(RavelTransformer.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    @staticmethod
    def f(x):
        return x.ravel()


class TahnEstimators(BaseEstimator, TransformerMixin):
    """
    refer
    https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
    https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python
    """

    def __init__(self):
        self.std_ = None
        self.mean_ = None
        self.n_seen_samples = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        return self

    def transform(self, X, copy=None):
        return 0.5 * (np.tanh(0.01 * (to_2d_array(X) - self.mean_) / self.std_) + 1)


class Features(object):
    # @staticmethod
    # def post_merge_feature_extraction(features, market_train_df):
    #     pipeline = UnionFeaturePipeline()
    #     # ## audience
    #     # In[ ]:
    #     # if FeatureSetting.max_shift_date > 0:
    #     #     lag_transformer = LagAggregationTransformer([3, 5, 10], shift_size=1,
    #     #                                                 remove_raw=FeatureSetting.remove_raw_for_lag,
    #     #                                                 scale=FeatureSetting.scale)
    #     #     market_train_df = lag_transformer.transform(market_train_df, n_pool=4)
    #     #     pipeline.add(lag_transformer)
    #
    #     if FeatureSetting.should_use_news_feature:
    #         news_feature_names = Features.extract_news_features(features, market_train_df, pipeline)
    #     else:
    #         news_feature_names = None
    #
    #     # market_train_df.to_csv("feature_df.csv")
    #     # In[ ]:
    #     # In[ ]:
    #
    #     return market_train_df, news_feature_names, pipeline

    # CONCATABLE_FEATURES = ["subjects", "audiences", "headline", "provider", "headlineTag"]
    #
    def __init__(self):
        # self.headlineTag_encoder = None
        # self.provider_encoder = None
        # self.audience_encoder = None
        # self.news = OrderedDict()
        # self.news_feature_names = None
        self.market_transformer = MarketFeatureTransformer()
        self.news_transformer = NewsFeatureTransformer()

    def fit(self, market_train_df: pd.DataFrame, news_train_df: pd.DataFrame):
        self.market_transformer.fit(market_train_df)
        self.news_transformer.fit(news_train_df)
        logger.info("feature fitting has done")
        return self

    def transform(self, market_train_df: pd.DataFrame, news_train_df: pd.DataFrame):
        logger.info("transforming into feature")
        return self.market_transformer.transform(market_train_df), self.news_transformer.transform(news_train_df)

    def fit_transform(self, market_train_df: pd.DataFrame, news_train_df: pd.DataFrame):
        return self.fit(market_train_df, news_train_df).transform(market_train_df, news_train_df)

    def get_linked_feature_matrix(self, link_df, market_indices=None):
        # print(link_df)
        link_df, news_feature_matrix = self.news_transformer.post_link_transform(link_df)
        # return sparse.hstack([self.market_transformer.feature_matrix, news_feature_matrix], dtype="float32",
        #                      format="csr")
        # return sparse.hstack([self.market_transformer.feature_matrix, news_feature_matrix], dtype="uint8",
        #                      format="csr")
        if market_indices is None and isinstance(link_df, pd.DataFrame):
            market_indices = link_df[MARKET_ID].tolist()

        return np.hstack([self.market_transformer.feature_matrix[market_indices], news_feature_matrix])

    def clear(self):
        self.market_transformer.clear()
        self.news_transformer.clear()

    def get_feature_num(self):
        return self.market_transformer.feature_matrix.shape[1] + self.news_transformer.feature_matrix.shape[1]


def log_object_sizes():
    for memory_info in ["{}: {}".format(v, sys.getsizeof(eval(v)) / 1000000000) for v in dir()]:
        logger.info(memory_info)


class FeatureTransformer(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, df):
        pass

    def fit(self, df):
        pass

    @abstractmethod
    def release_raw_field(self, df):
        pass

    def fit_transform(self, df):
        pass


class NullTransformer(FeatureTransformer):
    def transform(self, df):
        pass

    def release_raw_field(self, df):
        pass


class DfTransformer(FeatureTransformer):

    def transform(self, df):
        return df


class DropColumnsTransformer(NullTransformer):

    def __init__(self, columns):
        self.columns = columns

    def transform(self, df):
        df.drop(self.columns, axis=1, inplace=True)
        gc.collect()


class DateFilterTransformer(DfTransformer):

    def __init__(self, since_date, column="time"):
        self.since_date = since_date
        self.column = column

    def transform(self, df):
        df = df[df[self.column].dt.date >= self.since_date]
        return df

    def release_raw_field(self, df):
        pass


class LagAggregationTransformer(DfTransformer):
    LAG_FEATURES = ['returnsClosePrevMktres10', 'returnsClosePrevRaw10', 'open', 'close']

    def __init__(self, lags, shift_size, scale=True, remove_raw=False, n_pool=4):
        self.lags = lags
        self.shift_size = shift_size
        self.scale = scale
        if scale:
            self.scaler = None
        self.remove_raw = remove_raw
        self.imputer = None
        self.n_pool = n_pool

    @measure_time
    def transform(self, df, n_pool=None):
        if not n_pool:
            self.n_pool = n_pool

        df.sort_values(by="time", axis=0, inplace=True)
        logger.info("start extract lag...")
        group_features = [MARKET_ID, "time", "assetCode"] + self.LAG_FEATURES
        asset_code_groups = df[group_features].groupby("assetCode")
        asset_code_groups = [asset_code_group[1][group_features]
                             for asset_code_group in asset_code_groups]

        with Pool(self.n_pool) as pool:
            group_dfs = pool.map(self.extract_lag, asset_code_groups)
            group_dfs = pd.concat(group_dfs)
            group_dfs.drop(["time", "assetCode"] + self.LAG_FEATURES, axis=1, inplace=True)

            df = df.merge(group_dfs, how="left", copy=False, on=MARKET_ID)

            new_columns = list(itertools.chain.from_iterable(
                [['%s_lag_%s_mean' % (col, lag), '%s_lag_%s_max' % (col, lag), '%s_lag_%s_min' % (col, lag)]
                 for col, lag in itertools.product(self.LAG_FEATURES, self.lags)]))

        # df.drop(["time", "assetCode"], axis=1, inplace=True)

        # if self.scale:
        #     if not self.scaler:
        #         if FeatureSetting.scale_type == "minmax":
        #             self.scaler = {col: MinMaxScaler().fit(df[col].values.reshape((-1, 1))) for col in new_columns}
        #         elif FeatureSetting.scale_type == "standard":
        #             self.scaler = {col: StandardScaler().fit(df[col].values.reshape((-1, 1))) for col in new_columns}
        #
        #     for col in new_columns:
        #         df[col] = self.scaler[col].transform(df[col].values.reshape((-1, 1)))

        if self.remove_raw:
            df.drop(self.LAG_FEATURES, axis=1, inplace=True)

        # if self.imputer is None:
        #     self.imputer = {col: SimpleImputer(strategy="mean").fit(df[col].values.reshape((-1, 1))) for col in
        #                     new_columns}
        for col in new_columns:
            # print("imputing {}".format(col))
            # df[col] = self.imputer[col].transform(df[col].values.reshape((-1, 1)))
            df[col] = df[col].astype("float32")
        logger.info("Lag Aggregation has done")
        return df

    def extract_lag(self, asset_code_group):

        for col in self.LAG_FEATURES:
            for lag in self.lags:
                rolled = asset_code_group[col].shift(self.shift_size).rolling(window=lag)
                lag_mean = rolled.mean()
                lag_max = rolled.max()
                lag_min = rolled.min()
                # lag_std = rolled.std()
                asset_code_group['%s_lag_%s_mean' % (col, lag)] = lag_mean
                asset_code_group['%s_lag_%s_max' % (col, lag)] = lag_max
                asset_code_group['%s_lag_%s_min' % (col, lag)] = lag_min

        return asset_code_group

    def release_raw_field(self, df):
        pass

    def fit_transform(self, df):
        return self.transform(df)


class IdAppender(DfTransformer):

    def __init__(self, id_name):
        super().__init__()
        self.id_name = id_name

    def transform(self, df):
        df[self.id_name] = df.index.astype("int32")
        return df

    def release_raw_field(self, df):
        pass

    def fit_transform(self, df):
        return self.transform(df)


class ConfidenceAppender(DfTransformer):

    def transform(self, df):
        if NEXT_MKTRES_10 in df.columns:
            df["confidence"] = df[NEXT_MKTRES_10] >= 0
        return df

    def release_raw_field(self, df):
        pass

    def fit_transform(self, df):
        return self.transform(df)


# based on https://www.kaggle.com/qqgeogor/eda-script-67
class LimitMax(FunctionTransformer):
    def __init__(self, upper_limit,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.upper_limit = upper_limit
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.where(X >= self.upper_limit, X, self.upper_limit).reshape((-1, 1))


class NullTransformer(FunctionTransformer):
    def __init__(self, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return X


class WeekDayTransformer(FunctionTransformer):
    def __init__(self, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.cos(pd.Series(X).dt.dayofweek.values / 7).astype("float32").reshape((-1, 1))


class MonthTransformer(FunctionTransformer):
    def __init__(self, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.cos(pd.Series(X).dt.month.values / 12).astype("float32").reshape((-1, 1))


class DayTransformer(FunctionTransformer):
    def __init__(self, kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X, y=None):
        return np.cos(pd.Series(X).dt.day.values / 31).astype("float32").reshape((-1, 1))


class MarketFeatureTransformer(DfTransformer):
    # COLUMNS_NORMALIZED = ['volume', 'close', 'open',
    #                       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
    #                       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
    #                       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    # LOG_NORMAL_FIELDS = ['volume', 'close', 'open',
    #                      'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
    #                      'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
    #                      'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
    #                      'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    # LOG_NORMAL_FIELDS = []
    NUMERIC_COLUMNS = ['volume', 'close', 'open',
                       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    LAG_FEATURES = new_columns = list(itertools.chain.from_iterable(
        [['%s_lag_%s_mean' % (col, lag), '%s_lag_%s_max' % (col, lag), '%s_lag_%s_min' % (col, lag)]
         for col, lag in itertools.product(
            ['returnsClosePrevMktres10', 'returnsClosePrevRaw10', 'open', 'close'], [3, 5, 10])]))
    LABEL_OBJECT_FIELDS = ['assetName']
    DROP_COLS = ['universe', "returnsOpenNextMktres10"]
    TIME_COLS = ['time']

    def __init__(self):
        transformers = []

        # transformers.extend([
        #     (col,
        #      Pipeline([
        #          ("log", LogTransformer()),
        #          ("normalize", StandardScaler(copy=False)),
        #          ("fill_missing", SimpleImputer(strategy="median"))]),
        #      [col]) for col in set(self.COLUMNS_SCALED) & set(self.LOG_NORMAL_FIELDS)
        # ])

        scaled_columns = self.NUMERIC_COLUMNS
        if FeatureSetting.max_shift_date > 0:
            scaled_columns.extend(self.LAG_FEATURES)

        transformers.extend(
            [
                (col,
                 Pipeline([
                     # ("normalize", StandardScaler(copy=False)),
                     ("fill_missing", SimpleImputer(strategy="median"))]),
                 # ("discrete", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"))]),
                 [col]) for col in scaled_columns
            ]
        )

        transformers.extend(
            [
                ("time_week", WeekDayTransformer(), self.TIME_COLS[0]),
                ("time_month", MonthTransformer(), self.TIME_COLS[0]),
                ("time_day", DayTransformer(), self.TIME_COLS[0])
            ]
        )

        # transformers.extend(
        #     [(col + "bow",
        #       Pipeline([
        #           ("fill_missing", SimpleImputer(strategy="constant", fill_value="")),
        #           ('flatten', RavelTransformer()),
        #           ("encode", CountVectorizer(decode_error="ignore",
        #                                      stop_words=None,
        #                                      strip_accents="unicode",
        #                                      max_features=100,
        #                                      min_df=30,
        #                                      binary=True))
        #       ]), [col]) for col in self.LABEL_OBJECT_FIELDS]),

        self.encoder: ColumnTransformer = ColumnTransformer(transformers=transformers)
        self.feature_matrix = None

    def transform(self, df):
        self.feature_matrix = self.encoder.transform(df).astype("float32")
        # self.feature_matrix = self.feature_matrix
        self.release_raw_field(df)
        return df

    def fit(self, df):
        self.encoder.fit(df)
        return self

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def release_raw_field(self, df: pd.DataFrame):
        drop_cols = list \
            (set(self.NUMERIC_COLUMNS + self.LABEL_OBJECT_FIELDS))
        for col in self.DROP_COLS:
            if col in df.columns:
                drop_cols.append(col)

        df.drop(drop_cols, axis=1, inplace=True)
        gc.collect()

    def clear(self):
        self.feature_matrix = None
        gc.collect()


class NewsFeatureTransformer(DfTransformer):
    RAW_COLS = ['relevance', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 'marketCommentary']
    LOG_NORMAL_FIELDS = [
        'bodySize',
        'sentenceCount', 'wordCount',
        'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
        'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
        'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
        'volumeCounts7D']
    COLUMNS_SCALED = [
        'takeSequence',
        'bodySize', 'companyCount',
        'sentenceCount', 'wordCount',
        'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
        'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
        'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D',
        'volumeCounts7D']
    LABEL_COLS = ["sentimentClass", "provider", "urgency"]
    FIRST_MENTION_SENTENCE = "firstMentionSentence"
    BOW_COLS = ["headline"]
    MULTI_LABEL_COLS = ["subjects", "audiences"]

    LABEL_OBJECT_FIELDS = ['headlineTag']
    DROP_COLS = ['time', 'sourceId', 'sourceTimestamp', "assetName"]

    NUMERIC_COLS = list(set(RAW_COLS + LOG_NORMAL_FIELDS + COLUMNS_SCALED))
    NUMERIC_COL_INDICES = list(range(len(NUMERIC_COLS)))
    N_NUMERIC_COLS = len(NUMERIC_COLS)

    def __init__(self):
        transformers = []

        transformers.extend(
            [
                (
                    col,
                    "passthrough",
                    # KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"),
                    [col]
                ) for col in self.RAW_COLS
            ]
        )

        transformers.extend([
            (col,
             Pipeline([
                 ("log", LogTransformer()),
                 ("normalize", StandardScaler(copy=False)),
                 ("fill_missing", SimpleImputer(strategy="median"))]),
             # ("discrete", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"))]),
             [col]) for col in set(self.COLUMNS_SCALED) & set(self.LOG_NORMAL_FIELDS)
        ])

        transformers.extend(
            [
                (col,
                 Pipeline([
                     ("normalize", StandardScaler(copy=False)),
                     ("fill_missing", SimpleImputer(strategy="median"))]),
                 # ("discrete", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"))]),
                 [col]) for col in set(self.COLUMNS_SCALED) - set(self.LOG_NORMAL_FIELDS)
            ]
        )

        delay_transformers = []
        delay_transformers.extend([
            (col, OneHotEncoder(sparse=True, handle_unknown='ignore', dtype='uint8'), [col])
            for col in self.LABEL_COLS])
        delay_transformers.append(
            (self.FIRST_MENTION_SENTENCE,
             Pipeline(
                 [("limitMax", LimitMax(4)),
                  ("encoder", OneHotEncoder(sparse=True, handle_unknown='ignore', dtype='uint8'))]),
             [self.FIRST_MENTION_SENTENCE])
        )

        # transformers.extend(
        #     [(col + "bow",
        #       Pipeline([
        #           ("fill_missing", SimpleImputer(strategy="constant", fill_value="")),
        #           ('flatten', RavelTransformer()),
        #           ("encode", CountVectorizer(decode_error="ignore",
        #                                      stop_words="english",
        #                                      strip_accents="unicode",
        #                                      max_features=2000,
        #                                      binary=True, dtype="uint8"))
        #       ]), [col]) for col in self.BOW_COLS]),
        delay_transformers.extend(
            [(col, CountVectorizer(decode_error="ignore",
                                   strip_accents="unicode",
                                   min_df=5,
                                   max_features=2000,
                                   binary=True, dtype="uint8"), col)
             for col in self.MULTI_LABEL_COLS])

        delay_transformers.extend(
            [(col,
              Pipeline([
                  ("fill_missing", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                  ("encoder", OneHotEncoder(sparse=True, handle_unknown='ignore', dtype='uint8'))
              ]),
              [col])
             for col in self.LABEL_OBJECT_FIELDS]
        )

        self.encoder: ColumnTransformer = ColumnTransformer(transformers=transformers)
        self.delay_encoder: ColumnTransformer = ColumnTransformer(transformers=delay_transformers)
        self.feature_matrix = None
        self.store_df: pd.DataFrame = None
        self.n_delay_features = None

    def transform(self, df):
        self.feature_matrix = self.encoder.transform(df).astype("float32")
        # self.feature_matrix = self.feature_matrix.tocsr()
        self.store_df = df[
            self.LABEL_COLS + [self.FIRST_MENTION_SENTENCE] + self.MULTI_LABEL_COLS + self.LABEL_OBJECT_FIELDS]
        self.release_raw_field(df)
        return df

    def fit(self, df):
        self.encoder.fit(df)
        self.delay_encoder.fit(df)
        self.n_delay_features = self._get_delay_faeture_num()
        return self

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def release_raw_field(self, df):
        drop_cols = list \
            (set(self.RAW_COLS + [self.FIRST_MENTION_SENTENCE] + self.LABEL_COLS + self.MULTI_LABEL_COLS + self.BOW_COLS
                 + self.LOG_NORMAL_FIELDS + self.LABEL_OBJECT_FIELDS + self.COLUMNS_SCALED + self.DROP_COLS))
        df.drop(drop_cols, axis=1, inplace=True)
        gc.collect()

    def clear(self):
        self.feature_matrix = None
        self.store_df = None
        self.n_delay_features = None
        gc.collect()

    # @measure_time
    def aggregate(self, list_of_indices, pool_size=4, binary=True):
        self.encoded_cols_indices = list(range(self.N_NUMERIC_COLS, self.feature_matrix.shape[1]))
        # print(list_of_indices)
        # with Pool(pool_size) as pool:
        #     list_of_indices = pool.map(self.get_partial_agg, list_of_indices)
        list_of_indices = [self.get_partial_agg(indices) for indices in list_of_indices]
        # list_of_indices = sparse.csr_matrix(np.vstack(list_of_indices), dtype="float32")
        # list_of_indices = sparse.csr_matrix(np.vstack(list_of_indices), dtype="uint8")
        # print(list_of_indices[0].shape)
        # print(list_of_indices[1].shape)
        # print(list_of_indices)
        # list_of_indices = np.vstack(list_of_indices)
        list_of_indices = sparse.vstack(list_of_indices, dtype="float32", format="csr")
        # if binary:
        #     rows[rows != 0] = 1
        return list_of_indices

    def _get_delay_faeture_num(self):
        total = 0
        for transfomer_tuple in self.delay_encoder.transformers:
            transfomer = transfomer_tuple[1]
            if isinstance(transfomer, CountVectorizer):
                total += len(transfomer.vocabulary)
            elif isinstance(transfomer, OneHotEncoder):
                total += len(transfomer.categories)
            elif isinstance(transfomer, Pipeline):
                total += len(transfomer.named_steps["encoder"].categories)
        return total

    def get_partial_agg(self, ids):
        if not isinstance(ids, list) or np.isnan(ids[0]):
            # empty_feature = np.zeros((1, self.feature_matrix.shape[1] + ), dtype="float32")
            empty_feature = sparse.csr_matrix((1, self.feature_matrix.shape[1] + self.n_delay_features),
                                              dtype="float32")
            # empty_feature = np.zeros((1, self.feature_matrix.shape[1]), dtype="uint8")
            return empty_feature
        # numeric_partial = self.feature_matrix[ids][:, self.NUMERIC_COL_INDICES].mean(axis=0)
        # encoded_partial = self.feature_matrix[ids][:, self.encoded_cols_indices].sum(axis=0)
        # encoded_partial = self.feature_matrix[ids].sum(axis=0)
        # encoded_partial[encoded_partial != 0] = 1
        return sparse.hstack([self.feature_matrix[[int(id) for id in ids], :].mean(axis=0).reshape((1, -1)),
                              self.delay_encoder.transform(self.store_df.iloc[ids]).sum(axis=0).reshape((1, -1))],
                             dtype="float32")

    def post_link_transform(self, links):
        if isinstance(links, pd.DataFrame):
            aggregate_feature = self.aggregate(links[NEWS_ID].tolist())
            links.drop(NEWS_ID, axis=1, inplace=True)
        else:
            aggregate_feature = self.aggregate(links)
        return links, aggregate_feature


class TfDataGenerator(Sequence):

    def __init__(self, list_of_indices, features: Features, labels, batch_size=200):
        self.list_of_indices = list_of_indices
        # print(list_of_indices)
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.n_samples = len(self.list_of_indices)
        self.n_batches = self.n_samples // self.batch_size + int(bool(self.n_samples % self.batch_size))
        self._current_batch_num = 0

    # def __next__(self):
    #     while True:
    #         start = self._current_batch_num * self.batch_size
    #         if self._current_batch_num < self.n_batches - 1:
    #             end = (self._current_batch_num + 1) * self.batch_size
    #             yield self.features.get_linked_feature_matrix(self.list_of_indices[start:end]), self.labels[start:end]
    #             self._current_batch_num += 1
    #         else:
    #             yield self.features.get_linked_feature_matrix(self.list_of_indices[start:]), self.labels[start:]
    #             self._current_batch_num = 0

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        start = index * self.batch_size

        if index < self.n_batches - 1:
            end = (index + 1) * self.batch_size
            return self.features.get_linked_feature_matrix(
                self.list_of_indices[start:end], market_indices=list(range(start, end))), self.labels[start:end]
            # index += 1
        else:
            return self.features.get_linked_feature_matrix(self.list_of_indices[start:],
                                                           market_indices=list(
                                                               range(start, len(self.list_of_indices)))), \
                   self.labels[start:]
            # index = 0

    def on_epoch_end(self):
        pass


class SparseMLPWrapper(ModelWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_data_generator: TfDataGenerator = None
        self.valid_data_generator: TfDataGenerator = None

    def predict(self, x: Union[np.ndarray, sparse.spmatrix]):
        self.model = keras.models.load_model("mlp.model.h5")
        logger.info("predicting %d samples...".format(x.shape[0]))

        return self.model.predict(x)

    def train(self, sparse_input=False, **kwargs):
        input_ = keras.layers.Input(shape=(self.train_data_generator.features.get_feature_num(),), sparse=sparse_input,
                                    dtype="float32")
        x = keras.layers.Dense(192, activation='relu', kernel_initializer="he_normal",
                               kernel_regularizer=keras.regularizers.l1_l2(1e-4, 1e-3))(input_)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal",
                               kernel_regularizer=keras.regularizers.l1_l2(1e-4, 1e-3))(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation='relu', kernel_initializer="he_normal",
                               kernel_regularizer=keras.regularizers.l1_l2(1e-4, 1e-3))(x)
        x = keras.layers.Dropout(0.2)(x)
        output_ = keras.layers.Dense(1, activation="softmax", kernel_initializer='lecun_normal')(x)
        self.model = keras.Model(inputs=input_, outputs=output_)
        self.model.summary()

        checkpointer = ModelCheckpoint(filepath="mlp.model.h5",
                                       verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(lr=2e-2, decay=0.001),
            metrics=["accuracy"]
        )

        self.model.fit_generator(self.train_data_generator, self.train_data_generator.n_batches,
                                 epochs=50, validation_data=self.valid_data_generator,
                                 validation_steps=self.valid_data_generator.n_batches, verbose=0,
                                 callbacks=[checkpointer, early_stopping], shuffle=True)

    def create_dataset(self, market_train, features, train_batch_size, valid_batch_size):
        y, _ = ModelWrapper.to_x_y(market_train)
        # print(market_train[NEWS_ID])
        list_of_indices = market_train[NEWS_ID].tolist()
        list_of_indices, valid_indices, y, valid_y = ModelWrapper.split_train_validation(list_of_indices, y,
                                                                                         train_size=0.8)
        self.train_data_generator = TfDataGenerator(list_of_indices, features, y, batch_size=train_batch_size)
        self.valid_data_generator = TfDataGenerator(valid_indices, features, valid_y, batch_size=valid_batch_size)

        return None, None

    def clear(self):
        self.train_data_generator = None
        self.valid_data_generator = None
        gc.collect()


class Predictor(object):

    def __init__(self, linker, model, features, market_preprocess, news_preprocess):
        self.linker = linker
        self.model = model
        self.features: Features = features
        self.market_preprocess = market_preprocess
        self.news_preprocess = news_preprocess

    def predict_all(self, days, env):
        logger.info("=================prediction start ===============")

        stored_market_df = None
        stored_news_df = None
        max_time = None
        predict_start_id = 0

        def store_past_data(market_df, news_df, max_store_date=0):
            nonlocal stored_market_df
            nonlocal stored_news_df
            nonlocal predict_start_id
            if stored_market_df is None or max_store_date == 0:
                stored_market_df = market_df
                stored_news_df = news_df
                predict_start_id = 0
                return

            nonlocal max_time
            max_time = market_df["time"].max()

            min_time = max_time - offsets.Day(max_store_date)
            stored_market_df = stored_market_df[stored_market_df["time"] >= min_time]
            stored_news_df = stored_news_df[stored_news_df["firstCreated"] >= min_time]

            predict_start_id = len(stored_market_df)

            stored_market_df = pd.concat([stored_market_df, market_df], axis=0, ignore_index=True)
            stored_news_df = pd.concat([stored_news_df, news_df], axis=0, ignore_index=True)

        for (market_obs_df, news_obs_df, predictions_template_df) in tqdm(days):
            store_past_data(market_obs_df, news_obs_df, FeatureSetting.max_shift_date)
            market_obs_df_cp, news_obs_df_cp = stored_market_df.copy(), stored_news_df.copy()
            self.make_predictions(market_obs_df_cp, news_obs_df_cp, predictions_template_df, predict_start_id)
            env.predict(predictions_template_df)

    def make_predictions(self, market_obs_df, news_obs_df, predictions_df, predict_id_start):
        logger.info("predicting....")

        market_obs_df = self.market_preprocess.transform(market_obs_df)
        news_obs_df = self.news_preprocess.transform(news_obs_df)
        # print(market_obs_df[MARKET_ID])

        market_obs_df, news_obs_df = self.features.transform(market_obs_df, news_obs_df)

        if FeatureSetting.should_use_news_feature:
            self.linker.link(market_obs_df, news_obs_df)
            market_obs_df = self.linker.create_new_market_df()
            self.linker.clear()
            # print(market_obs_df[MARKET_ID])
        del news_obs_df
        gc.collect()

        feature_matrix = self.features.get_linked_feature_matrix(market_obs_df)

        logger.info("input size: {}".format(feature_matrix.shape))
        predictions = self.model.predict(feature_matrix)
        predict_indices = market_obs_df[MARKET_ID][market_obs_df[MARKET_ID] >= predict_id_start].astype("int").tolist()

        logger.info("predicted size: {}".format(predictions.shape))
        # logger.info("predicted indices: {}".format(predict_indices))
        # print(predict_indices)
        predictions = predictions[predict_indices] * 2 - 1
        predictions = predictions[np.argsort(predict_indices)]
        logger.info("predicted size: {}".format(predictions.shape))
        logger.info("predicted target size: {}".format(predictions_df.shape))
        predictions_df.confidenceValue = predictions
        logger.info("prediction done")


if __name__ == '__main__':
    logger = logging.getLogger("root")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler("main.log"))
    try:
        main()
    except RuntimeError as e:
        logger.exception("runtime error %s", e)
