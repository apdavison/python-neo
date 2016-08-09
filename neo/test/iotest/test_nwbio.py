# -*- coding: utf-8 -*-
"""
Tests of neo.io.nwbio
"""

# needed for python 3 compatibility
from __future__ import division

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest
try:
    import nwb
    HAVE_NWB = True
except ImportError:
    HAVE_NWB = False
from neo.io import NWBIO
from neo.test.iotest.common_io_test import BaseTestIO

@unittest.skipUnless(HAVE_NWB, "requires nwb")
class TestNWBIO(BaseTestIO, unittest.TestCase):
    ioclass = NWBIO
    files_to_test = []
    files_to_download =  []


if __name__ == "__main__":
    unittest.main()
