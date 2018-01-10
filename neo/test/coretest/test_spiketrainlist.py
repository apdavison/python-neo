# -*- coding: utf-8 -*-
"""
Tests of the neo.core.spiketrainlist.SpikeTrainList class
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import quantities as pq

from neo.core.spiketrain import SpikeTrain
from neo.core.spiketrainlist import SpikeTrainList


class TestSpikeTrainList(unittest.TestCase):

    def setUp(self):
        pass
