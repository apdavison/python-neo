#
"""
Tests of neo.io.nwbio
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment, Unit, Block, ChannelIndex, ImageSequence
try:
    import pynwb
    from neo.io.nwbio import NWBIO
    HAVE_PYNWB = True
except (ImportError, SyntaxError):
    NWBIO = None
    HAVE_PYNWB = False
import quantities as pq
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from neo.test.rawiotest.tools import create_local_temp_dir


@unittest.skipUnless(HAVE_PYNWB, "requires pynwb")
class TestNWBIO(unittest.TestCase):
    ioclass = NWBIO
    files_to_download = [
        #        Files from Allen Institute :
        # NWB 1
        # "http://download.alleninstitute.org/informatics-archive/prerelease/H19.28.012.11.05-2.nwb",  # 64 MB
        # "http://download.alleninstitute.org/informatics-archive/prerelease/H19.29.141.11.21.01.nwb",  # 7 MB
        #  "http://download.alleninstitute.org/informatics-archive/prerelease/H19.28.012.11.05-3.nwb", # 85 MB
        #  "http://download.alleninstitute.org/informatics-archive/prerelease/H19.28.012.11.05-4.nwb", # 72 MB
        ###  "http://download.alleninstitute.org/informatics-archive/prerelease/behavior_ophys_session_775614751.nwb", # 808 MB "'AIBS_ophys_behavior' not a namespace"
                ##  "http://download.alleninstitute.org/informatics-archive/prerelease/behavior_ophys_session_778644591.nwb", # 1,1 GB
                ##  "http://download.alleninstitute.org/informatics-archive/prerelease/behavior_ophys_session_783927872.nwb", # 1,4 GB
                ##  "http://download.alleninstitute.org/informatics-archive/prerelease/behavior_ophys_session_783928214.nwb", # 1,5 GB
                ##  "http://download.alleninstitute.org/informatics-archive/prerelease/behavior_ophys_session_784482326.nwb", # 1,1 GB
        
        # Compressed files
        ##  "http://download.alleninstitute.org/informatics-archive/prerelease/ecephys_session_715093703.nwb.bz2", # 861 MB
        ####  "http://download.alleninstitute.org/informatics-archive/prerelease/ecephys_session_759228117.nwb.bz2", # 643 MB
        ####  "http://download.alleninstitute.org/informatics-archive/prerelease/ecephys_session_759228117.nwb", # 643 MB Error 404
        ##  "http://download.alleninstitute.org/informatics-archive/prerelease/ecephys_session_764437248.nwb.bz2", # 704 MB
        ##  "http://download.alleninstitute.org/informatics-archive/prerelease/ecephys_session_785402239.nwb.bz2", # 577 MB
        
        # NWB 2 :   
        ###  "http://download.alleninstitute.org/informatics-archive/prerelease/pxp_examples_for_nwb_2/Oldest_published_data/Pvalb-IRES-Cre%3bAi14(IVSCC)-165172.05.02-compressed-V1.nwb", # 147 MB (no data_type found for builder root)
        #  "http://download.alleninstitute.org/informatics-archive/prerelease/pxp_examples_for_nwb_2/Oldest_published_data/Pvalb-IRES-Cre%3bAi14(IVSCC)-165172.05.02-compressed-V2.nwb", # 162 MB
        ###  "http://download.alleninstitute.org/informatics-archive/prerelease/pxp_examples_for_nwb_2/Patch_seq_v1/Vip-IRES-Cre%3bAi14-331294.04.01.01-compressed-V1.nwb", # 7,1 MB (no data_type found for builder root)
        #  "http://download.alleninstitute.org/informatics-archive/prerelease/pxp_examples_for_nwb_2/Patch_seq_v1/Vip-IRES-Cre%3bAi14-331294.04.01.01-compressed-V2.nwb", # 17 MB    
        ###  "http://download.alleninstitute.org/informatics-archive/prerelease/pxp_examples_for_nwb_2/Patch_seq_v2/Ctgf-T2A-dgCre%3bAi14-495723.05.02.01-compressed-V1.nwb", # 9,6 MB (no data_type found for builder root)
        #  "http://download.alleninstitute.org/informatics-archive/prerelease/pxp_examples_for_nwb_2/Patch_seq_v2/Ctgf-T2A-dgCre%3bAi14-495723.05.02.01-compressed-V2.nwb", # 21 MB
       
        #        Files from Steinmetz et al. Nature 2019 : 
        ###  "https://ndownloader.figshare.com/files/19903865", # Steinmetz2019_Cori_2016-12-14.nwb # 311,8 MB
        
        #        Files from Buzsaki Lab
        # Corrupted files
        ###  "https://buzsakilab.nyumc.org/datasets/NWB/SenzaiNeuron2017/YutaMouse20/YutaMouse20-140328.nwb", # 445,6 MB (Error : bad object header version number)
        ###  "https://buzsakilab.nyumc.org/datasets/NWB/SenzaiNeuron2017/YutaMouse55/YutaMouse55-160910.nwb", # 461,1 MB (Error : bad object header version number)
        ###  "https://buzsakilab.nyumc.org/datasets/NWB/SenzaiNeuron2017/YutaMouse57/YutaMouse57-161011.nwb", # 88,4 MB (Error : bad object header version number)

        #       Files from Svoboda Lab
        #       Files extracted from the paper Chen et al Neuron 2017
        ###  "https://www.dropbox.com/sh/i5kqq99wq4qbr5o/AACE5R4THCXYEbEZpsFtPGQpa/nwb2/tsai_wen_nwb2/nwb_an041_20140821_vM1_180um.nwb?dl=0", # Corrupted file

        #       Files from PyNWB Test Data
        "/Users/legouee/NWBwork/my_notebook/neo_test.nwb" # Issue 796
        ##       Zip files to download from "https://drive.google.com/drive/folders/1g1CpnoMd9s9L-sHBWVyklp3-xJcLGeFt"
        #       Local files
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/ecephys_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/ophys_example.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_multicontainerinterface.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/icephys_example.nwb", # OK
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/cache_spec_example.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_sparse_iterwrite_example.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_iterwrite_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/external_linkdataset_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/external_linkcontainer_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/external2_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/external1_example.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/example_file_path.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_sparse_iterwrite_multifile.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_sparse_iterwrite_largechunks_example.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_sparse_iterwrite_largechunks_compressed_example.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_sparse_iterwrite_largearray.nwb",
######            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/basic_sparse_iterwrite_compressed_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/advanced_io_example.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_timestamps_linking.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_append.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_TimeSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_PatchClampSeries.nwb", # OK
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_LFP.nwb", # Ok
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_IntracellularElectrode.nwb", # Ok
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_FilteredEphys.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_FeatureExtraction.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_EventWaveform.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_EventDetection.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_ElectrodeGroup.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_ElectricalSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_DynamicTable.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_Device.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_CurrentClampStimulusSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_Clustering.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_ClusterWaveforms.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_time_series_modular_link.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_time_series_modular_data.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_VoltageClampStimulusSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_VoltageClampSeries.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_Units.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_SweepTable.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_Subject.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_IZeroClampSeries.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_DecompositionSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_CurrentClampSeries.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_TwoPhotonSeries.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_TimeIntervals.nwb",
###            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_RoiResponseSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_PlaneSegmentation.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_OptogeneticStimulusSite.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_OptogeneticSeries.nwb",
#            "/Users/legouee/Desktop/NWB/NWB_files/PyNWB_Test_Data/reference_nwb_files_5073997e25b306c7395f6ceebdaf4f7af066ffef/test_ImagingPlane.nwb",



        #       Local files
        #       Files created from NWB_N_Tutorial_Extracellular_Electrophysiology_Janelia_2019
        ###   "/Users/legouee/NWBwork/my_notebook/NWB_data_from_Labs/test_ephys.nwb",

        ## "/Users/legouee/Desktop/NWB/NWB_files/Allen_Institute/ecephys_session_785402239.nwb",
        # "/Users/legouee/Desktop/NWB/NWB_files/Allen_Institute/H19.29.141.11.21.01.nwb", # 7 MB
        ## "/Users/legouee/Desktop/NWB/NWB_files/Allen_Institute/Ctgf-T2A-dgCre;Ai14-495723.05.02.01-compressed-V2.nwb", # 22 MB
        ##"/Users/legouee/Desktop/NWB/NWB_files/Allen_Institute/Ctgf-T2A-dgCre;Ai14-495723.05.02.01-compressed-V1.nwb", # 10 MB
        # "/Users/legouee/Desktop/NWB/NWB_files/Allen_Institute/H19.28.012.11.05-2.nwb",
        #### "/Users/legouee/Desktop/NWB/NWB_files/Example.nwb", # Extract from NWB Github Issue 1077 for builder root

    ]

    def test_read(self):
        self.local_test_dir = create_local_temp_dir("nwb")
        os.makedirs(self.local_test_dir, exist_ok=True)
        print("self.files_to_download[0] = ", self.files_to_download[0])

#        for url in self.files_to_download:
#            local_filename = os.path.join(self.local_test_dir, url.split("/")[-1])
#            print("local_filename = ", local_filename)
#            print("self.local_test_dir = ", self.local_test_dir)
#
#            if not os.path.exists(local_filename):
#                try:
####                    urlretrieve(url, self.local_filename[0])
#                    urlretrieve(url, local_filename) # Original
#                except IOError as exc:
#                    raise unittest.TestCase.failureException(exc)
#            io = NWBIO(local_filename, 'r')
#            blocks = io.read()

        io = NWBIO(self.files_to_download[0], 'r')
        blocks = io.read()
    
    def test_roundtrip(self):

        # Define Neo blocks
        bl0 = Block(name='First block')
        bl1 = Block(name='Second block')
        bl2 = Block(name='Third block')
        original_blocks = [bl0, bl1, bl2]

        num_seg = 4  # number of segments
        num_chan = 3  # number of channels

        for blk in original_blocks:

            for ind in range(num_seg):  # number of Segment
                seg = Segment(index=ind)
                seg.block = blk
                blk.segments.append(seg)

            for seg in blk.segments:  # AnalogSignal objects

                # 3 Neo AnalogSignals
                a = AnalogSignal(np.random.randn(44, num_chan) * pq.nA,
                                 sampling_rate=10 * pq.kHz,
                                 t_start=50 * pq.ms)
                b = AnalogSignal(np.random.randn(64, num_chan) * pq.mV,
                                 sampling_rate=8 * pq.kHz,
                                 t_start=40 * pq.ms)
                c = AnalogSignal(np.random.randn(33, num_chan) * pq.uA,
                                 sampling_rate=10 * pq.kHz,
                                 t_start=120 * pq.ms)

                # 2 Neo IrregularlySampledSignals
                d = IrregularlySampledSignal(np.arange(7.0)*pq.ms,
                                             np.random.randn(7, num_chan)*pq.mV)

                # 2 Neo SpikeTrains
                train = SpikeTrain(times=[1, 2, 3] * pq.s, t_start=1.0, t_stop=10.0)
                train2 = SpikeTrain(times=[4, 5, 6] * pq.s, t_stop=10.0)
                # todo: add waveforms

                # 1 Neo Event
                evt = Event(times=np.arange(0, 30, 10) * pq.ms,
                            labels=np.array(['ev0', 'ev1', 'ev2']))

                # 2 Neo Epochs
                epc = Epoch(times=np.arange(0, 30, 10) * pq.s,
                            durations=[10, 5, 7] * pq.ms,
                            labels=np.array(['btn0', 'btn1', 'btn2']))

                epc2 = Epoch(times=np.arange(10, 40, 10) * pq.s,
                             durations=[9, 3, 8] * pq.ms,
                             labels=np.array(['btn3', 'btn4', 'btn5']))

                seg.spiketrains.append(train)
                seg.spiketrains.append(train2)

                seg.epochs.append(epc)
                seg.epochs.append(epc2)

                seg.analogsignals.append(a)
                seg.analogsignals.append(b)
                seg.analogsignals.append(c)
                seg.irregularlysampledsignals.append(d)
                seg.events.append(evt)
                a.segment = seg
                b.segment = seg
                c.segment = seg
                d.segment = seg
                evt.segment = seg
                train.segment = seg
                train2.segment = seg
                epc.segment = seg
                epc2.segment = seg

        # write to file
        test_file_name = "test_round_trip.nwb"
        iow = NWBIO(filename=test_file_name, mode='w')
        iow.write_all_blocks(original_blocks)

        ior = NWBIO(filename=test_file_name, mode='r')
        retrieved_blocks = ior.read_all_blocks()

        self.assertEqual(len(retrieved_blocks), 3) ######
        self.assertEqual(len(retrieved_blocks[2].segments), num_seg)

        original_signal_22b = original_blocks[2].segments[2].analogsignals[1]
        print("original_signal_22b = ", original_signal_22b)
        print("original_blocks[2].segments[2] = ", original_blocks[2].segments[2])
        print("retrieved_blocks[2].segments[2] = ", retrieved_blocks[2].segments[2])
        #retrieved_signal_22b = retrieved_blocks[2].segments[2].analogsignals[1]
        for attr_name in ("name", "units", "sampling_rate", "t_start"):
        #    retrieved_attribute = getattr(retrieved_signal_22b, attr_name)
            original_attribute = getattr(original_signal_22b, attr_name)
        #    self.assertEqual(retrieved_attribute, original_attribute)
        #assert_array_equal(retrieved_signal_22b.magnitude, original_signal_22b.magnitude)

        original_issignal_22d = original_blocks[2].segments[2].irregularlysampledsignals[0]
        #retrieved_issignal_22d = retrieved_blocks[2].segments[2].irregularlysampledsignals[0]
        for attr_name in ("name", "units", "t_start"):
        #    retrieved_attribute = getattr(retrieved_issignal_22d, attr_name)
            original_attribute = getattr(original_issignal_22d, attr_name)
        #    self.assertEqual(retrieved_attribute, original_attribute)
        #assert_array_equal(retrieved_issignal_22d.times.rescale('ms').magnitude,
        #                   original_issignal_22d.times.rescale('ms').magnitude)
        #assert_array_equal(retrieved_issignal_22d.magnitude, original_issignal_22d.magnitude)

        original_event_11 = original_blocks[1].segments[1].events[0]
        retrieved_event_11 = retrieved_blocks[1].segments[1].events[0]
        for attr_name in ("name",):
            retrieved_attribute = getattr(retrieved_event_11, attr_name)
            original_attribute = getattr(original_event_11, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_event_11.rescale('ms').magnitude,
                           original_event_11.rescale('ms').magnitude)
        assert_array_equal(retrieved_event_11.labels, original_event_11.labels)

        original_spiketrain_131 = original_blocks[1].segments[1].spiketrains[1]
        #retrieved_spiketrain_131 = retrieved_blocks[1].segments[1].spiketrains[1]
        for attr_name in ("name", "t_start", "t_stop"):
            #retrieved_attribute = getattr(retrieved_spiketrain_131, attr_name)
            original_attribute = getattr(original_spiketrain_131, attr_name)
            #self.assertEqual(retrieved_attribute, original_attribute)
        #assert_array_equal(retrieved_spiketrain_131.times.rescale('ms').magnitude,
        #                   original_spiketrain_131.times.rescale('ms').magnitude)

        original_epoch_11 = original_blocks[1].segments[1].epochs[0]
        #retrieved_epoch_11 = retrieved_blocks[1].segments[1].epochs[0]
        for attr_name in ("name",):
        #    retrieved_attribute = getattr(retrieved_epoch_11, attr_name)
            original_attribute = getattr(original_epoch_11, attr_name)
        #    self.assertEqual(retrieved_attribute, original_attribute)
        #assert_array_equal(retrieved_epoch_11.rescale('ms').magnitude,
        #                   original_epoch_11.rescale('ms').magnitude)
        #assert_allclose(retrieved_epoch_11.durations.rescale('ms').magnitude,
        #                original_epoch_11.durations.rescale('ms').magnitude)
        #assert_array_equal(retrieved_epoch_11.labels, original_epoch_11.labels)


if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()
