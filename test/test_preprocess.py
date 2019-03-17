import logging
import sys
from unittest import TestCase

import numpy as np
import pandas as pd

from not_final_kernels.final_local_but_oom_kernel import NewsPreprocess, load_train_dfs, MarketPreprocess, \
    TahnEstimators

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TestPreprocess(TestCase):

    def test_NewsPreprocess(self):
        sut = NewsPreprocess()
        _, _, newsdf = load_train_dfs()
        processed_newsdf = sut.fit_transform(newsdf)
        print(newsdf.head())

    def test_MarketPreprocess(self):
        sut = MarketPreprocess()
        _, maket_df, _ = load_train_dfs()
        sut.fit_transform(maket_df)
        print(maket_df.head())


class TestTahnEstimators(TestCase):
    RANDOM_SEED = 10

    def test_fit_transform(self):
        np.random.seed(self.RANDOM_SEED)
        seq = pd.Series(np.arange(0, 10))
        sut = TahnEstimators()
        result = sut.fit_transform(seq)
        logger.info(result)
