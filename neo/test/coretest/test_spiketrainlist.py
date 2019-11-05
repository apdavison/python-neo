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

    def test_create_from_spiketrain_array(self):
        spike_time_array = np.array([0.5, 0.6, 0.7, 1.1, 11.2, 23.6, 88.5, 99.2])
        channel_id_array = np.array([0, 0, 1, 2, 1, 0, 2, 0])
        all_channel_ids = (0, 1, 2, 3)
        stl = SpikeTrainList.from_spike_time_array(spike_time_array,
                                                   channel_id_array,
                                                   all_channel_ids=all_channel_ids,
                                                   units='ms',
                                                   t_start=0*pq.ms,
                                                   t_stop=100.0*pq.ms)
        as_list = list(stl)
        assert_array_equal(as_list[0].times.magnitude,
                           np.array([0.5, 0.6, 23.6, 99.2]))
        assert_array_equal(as_list[1].times.magnitude,
                           np.array([ 0.7, 11.2]))
        assert_array_equal(as_list[2].times.magnitude,
                           np.array([ 1.1, 88.5]))
        assert_array_equal(as_list[3].times.magnitude,
                           np.array([]))
