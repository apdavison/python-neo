"""
Tests of neo.io.hdf5io_new

"""

import unittest
import sys
import numpy as np
from numpy.testing import assert_array_equal
from quantities import kHz, mV, ms, second, nA

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
from neo.io.hdf5io import NeoHdf5IO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path


@unittest.skipUnless(HAVE_H5PY, "requires h5py")
class ReadOldNeoHdf5IOTest(BaseTestIO, unittest.TestCase):
    """
    Test that data generated by NeoHdf5IO in Neo versions 0.3, 0.4 are
    read correctly.
    """

    ioclass = NeoHdf5IO
    files_to_test = ["neo_hdf5_example.h5"]
    files_to_download = files_to_test

    def test_read_with_merge(self):
        test_file = get_test_file_full_path(
            self.ioclass,
            filename=self.files_to_test[0],
            directory=self.local_test_dir,
            clean=False,
        )
        io = NeoHdf5IO(test_file)
        blocks = io.read_all_blocks(merge_singles=True)

        # general tests, true for both blocks
        for block in blocks:
            for segment in block.segments:
                self.assertEqual(segment.block, block)

        # tests of Block #1, which is constructed from "array" (multi-channel)
        # objects, so should be straightforward to convert to the version 0.5 API
        block0 = blocks[0]
        self.assertEqual(block0.name, "block1")
        self.assertEqual(block0.index, 1234)
        self.assertEqual(block0.annotations["foo"], "bar")
        self.assertEqual(len(block0.segments), 3)
        for segment in block0.segments:
            self.assertEqual(len(segment.analogsignals), 2)
            as0 = segment.analogsignals[0]
            self.assertEqual(as0.shape, (1000, 4))
            self.assertEqual(as0.sampling_rate, 1 * kHz)
            self.assertEqual(as0.units, mV)
            self.assertEqual(as0.segment, segment)

            self.assertEqual(len(segment.spiketrains), 4)
            st = segment.spiketrains[-1]
            self.assertEqual(st.units, ms)
            self.assertEqual(st.t_stop, 1000 * ms)
            self.assertEqual(st.t_start, 0 * ms)
            self.assertEqual(st.segment, segment)

            self.assertEqual(len(segment.events), 1)
            ev = segment.events[0]
            assert_array_equal(
                ev.labels,
                np.array(
                    ["trig0", "trig1", "trig2"],
                    dtype=(sys.byteorder == "little" and "<" or ">") + "U5",
                ),
            )
            self.assertEqual(ev.units, second)
            assert_array_equal(ev.magnitude, np.arange(0, 30, 10))
            self.assertEqual(ev.segment, segment)

            self.assertEqual(len(segment.epochs), 1)
            ep = segment.epochs[0]
            assert_array_equal(
                ep.labels,
                np.array(
                    ["btn0", "btn1", "btn2"],
                    dtype=(sys.byteorder == "little" and "<" or ">") + "U4",
                ),
            )
            assert_array_equal(ep.durations.magnitude, np.array([10, 5, 7]))
            self.assertEqual(ep.units, second)
            assert_array_equal(ep.magnitude, np.arange(0, 30, 10))
            self.assertEqual(ep.segment, segment)

            self.assertEqual(len(segment.irregularlysampledsignals), 2)
            iss0 = segment.irregularlysampledsignals[0]
            self.assertEqual(iss0.shape, (3, 2))
            assert_array_equal(iss0.times, [0.01, 0.03, 0.12] * second)
            assert_array_equal(iss0.magnitude, np.array([[4, 3], [5, 4], [6, 3]]))
            self.assertEqual(iss0.units, nA)
            self.assertEqual(iss0.segment, segment)

            iss1 = segment.irregularlysampledsignals[1]
            self.assertEqual(iss1.shape, (3, 1))
            assert_array_equal(iss1.times, [0.02, 0.05, 0.15] * second)
            self.assertEqual(iss1.units, nA)
            assert_array_equal(iss1.magnitude, np.array([[3], [4], [3]]))

        # tests of Block #2, which is constructed from "singleton"
        # (single-channel) objects, so is potentially tricky to convert to the
        # version 0.5 API
        block1 = blocks[1]
        self.assertEqual(block1.name, "block2")

        for segment in block1.segments:
            self.assertEqual(len(segment.analogsignals), 2)
            as0 = segment.analogsignals[0]
            self.assertEqual(as0.shape, (1000, 4))
            self.assertEqual(as0.sampling_rate, 1 * kHz)
            self.assertEqual(as0.units, mV)
            self.assertEqual(as0.segment, segment)

            self.assertEqual(len(segment.spiketrains), 7)
            st = segment.spiketrains[-1]
            self.assertEqual(st.units, ms)
            self.assertEqual(st.t_stop, 1000 * ms)
            self.assertEqual(st.t_start, 0 * ms)
            self.assertEqual(st.segment, segment)

            self.assertEqual(len(segment.events), 0)
            self.assertEqual(len(segment.epochs), 0)

        self.assertEqual(len(block1.channel_indexes), 3)

        ci0 = block1.channel_indexes[0]
        self.assertEqual(ci0.name, "electrode1")
        self.assertEqual(len(ci0.analogsignals), 1)
        as00 = ci0.analogsignals[0]
        self.assertEqual(as00.segment, segment)
        self.assertEqual(as00.shape, (1000, 4))
        self.assertEqual(id(as00), id(segment.analogsignals[0]))
        self.assertEqual(as00.mean(), segment.analogsignals[0].mean())
        self.assertEqual(as00.channel_index, ci0)
        assert_array_equal(ci0.index, np.array([0, 1, 2, 3]))
        assert_array_equal(ci0.channel_ids, np.array([0, 1, 2, 3]))
        self.assertEqual(len(ci0.units), 2)
        self.assertEqual(len(ci0.units[0].spiketrains), 2)
        self.assertEqual(id(ci0.units[0].spiketrains[0]), id(block1.segments[0].spiketrains[0]))
        self.assertEqual(id(ci0.units[0].spiketrains[1]), id(block1.segments[1].spiketrains[0]))
        self.assertEqual(id(ci0.units[1].spiketrains[0]), id(block1.segments[0].spiketrains[1]))

        ci1 = block1.channel_indexes[1]
        self.assertEqual(ci1.name, "electrode2")
        self.assertEqual(len(ci1.analogsignals), 1)
        as10 = ci1.analogsignals[0]
        self.assertEqual(as10.segment, segment)
        self.assertEqual(as10.shape, (1000, 4))
        self.assertEqual(id(as10), id(segment.analogsignals[1]))
        self.assertEqual(as10.mean(), segment.analogsignals[1].mean())
        self.assertEqual(as10.channel_index, ci1)
        assert_array_equal(ci1.index, np.array([0, 1, 2, 3]))
        assert_array_equal(ci1.channel_ids, np.array([4, 5, 6, 7]))
        self.assertEqual(len(ci1.units), 5)
        self.assertEqual(id(ci1.units[0].spiketrains[0]), id(block1.segments[0].spiketrains[2]))
        self.assertEqual(id(ci1.units[3].spiketrains[1]), id(block1.segments[1].spiketrains[5]))

        ci2 = block1.channel_indexes[2]
        self.assertEqual(ci2.name, "my_favourite_channels")
        self.assertEqual(len(ci2.analogsignals), 1)
        self.assertEqual(id(ci2.analogsignals[0]), id(as00))
        assert_array_equal(ci2.index, np.array([1, 3]))
        assert_array_equal(ci2.channel_ids, np.array([1, 3]))
