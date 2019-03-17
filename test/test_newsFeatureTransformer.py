from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from not_final_kernels.final_local_but_oom_kernel import NewsFeatureTransformer, MarketFeatureTransformer

# from main import NewsPreprocess

TEST_MARKET_DATA = Path(__file__).parent.joinpath("../data/test/marketdata_sample.csv")
TEST_NEWS_DATA = Path(__file__).parent.joinpath("../data/test/news_sample.csv")


class TestNewsFeatureTransformer(TestCase):
    def test_fit_transform(self):
        df = pd.read_csv(TEST_NEWS_DATA, encoding="utf-8", engine="python")

        sut = NewsFeatureTransformer()
        new_df = sut.fit_transform(df)

        dense_feature = sut.feature_matrix.todense()
        print(dense_feature)
        print(sut.encoder.named_transformers_.get("subjects").vocabulary_)
        # TODO assertion

    def test_aggregate(self):
        df = pd.read_csv(TEST_NEWS_DATA, encoding="utf-8", engine="python")

        sut = NewsFeatureTransformer()
        new_df = sut.fit_transform(df)

        result = sut.aggregate([[1, 2, 3], [4, 5, 6, 7], np.NaN])

        print(result.todense())


class TestMarketFeatureTransformer(TestCase):
    def test_fit_transform(self):
        df = pd.read_csv(TEST_MARKET_DATA, encoding="utf-8", engine="python")

        sut = MarketFeatureTransformer()
        market_df = sut.fit_transform(df)

        print(sut.feature_matrix)
        # print(sut.encoder.named_transformers_.get("subjects").vocabulary_)
        # TODO assertion
