import logging
import sys
from unittest import TestCase

from not_final_kernels.final_local_but_oom_kernel import measure_time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TestMeasure_time(TestCase):
    # TODO assertion
    def test_measure_time(self):
        @measure_time
        def repeat_add(n):
            total = 0
            for i in range(n):
                total += i
            return total

        total = repeat_add(1000000)
        logger.info("%d", total)
