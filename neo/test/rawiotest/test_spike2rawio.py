import unittest

from neo.rawio.spike2rawio import Spike2RawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestSpike2RawIO(
    BaseTestRawIO, unittest.TestCase,
):
    rawioclass = Spike2RawIO
    files_to_download = [
        "File_spike2_1.smr",
        "File_spike2_2.smr",
        "File_spike2_3.smr",
        "130322-1LY.smr",  # this is for bug 182
        "multi_sampling.smr",  # this is for bug 466
        "Two-mice-bigfile-test000.smr",  # SONv9 file
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
