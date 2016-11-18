"""
This script generates an HDF5 file using the NeoHdf5IO, for the purpose
of testing conversion from Neo 0.3/0.4 to 0.5.

"""
from datetime import datetime
from itertools import chain
import numpy as np
import neo
from quantities import mV, ms, kHz, s, nA

blocks = [neo.Block(name="block1",
                    file_datetime=datetime.now(),
                    index=1234,
                    foo="bar"),
          ]

# Block 1: arrays
segments0 = [neo.Segment(name="seg1{0}".format(i))
             for i in range(1, 4)]
blocks[0].segments = segments0
for j, segment in enumerate(segments0):
    segment.block = blocks[0]
    segment.analogsignals = [neo.AnalogSignal(
                                        np.random.normal(-60.0 + j + i, 10.0, size=(1000, 4)),
                                        units=mV,
                                        sampling_rate=1*kHz
                                  ) for i in range(2)]
    segment.spiketrains = [neo.SpikeTrain(np.arange(100 + j + i, 900, 10.0), t_stop=1000*ms, units=ms)
                           for i in range(4)]
    # todo: add spike waveforms
    segment.events = [neo.Event(np.arange(0, 30, 10)*s,
                                          labels=np.array(['trig0', 'trig1', 'trig2']))]
    segment.epochs = [neo.Epoch(times=np.arange(0, 30, 10)*s,
                                          durations=[10, 5, 7]*ms,
                                          labels=np.array(['btn0', 'btn1', 'btn2']))]
    segment.irregularlysampledsignals = [neo.IrregularlySampledSignal([0.01, 0.03, 0.12]*s, [4, 5, 6]*nA),
                                         neo.IrregularlySampledSignal([0.01, 0.03, 0.12]*s, [3, 4, 3]*nA),
                                         neo.IrregularlySampledSignal([0.02, 0.05, 0.15]*s, [3, 4, 3]*nA)]
    for obj in chain(segment.analogsignals, segment.analogsignals, segment.events,
                     segment.epochs, segment.irregularlysampledsignals):
        obj.segment = segment



io = neo.io.NWBIO("neo_nwb_example1.nwb")
io.write_block(blocks[0])

