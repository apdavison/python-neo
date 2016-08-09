# -*- coding: utf-8 -*-
"""
Class for reading data from a Neurodata Without Borders (NWB) dataset

Depends on: h5py, nwb, dateutil

Supported: Read, Write

Author: Andrew P. Davison, CNRS

"""

from __future__ import absolute_import
from __future__ import division
from itertools import chain
import shutil
import tempfile
from datetime import datetime
from os.path import join
import dateutil.parser
import numpy as np

try:
    import h5py
    from nwb import nwb_file
    from nwb import nwb_utils
    HAVE_NWB = True
except ImportError as err:
    HAVE_NWB = False

import quantities as pq
from neo.io.baseio import BaseIO
from neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block)

neo_extension = {"fs": {"neo": {
    "info": {
        "name": "Neo TimeSeries extension",
        "version": "0.1",
        "date": "2016-08-05",
        "author": "Andrew P. Davison",
        "contact": "andrew.davison@unic.cnrs-gif.fr",
        "description": ("Extension defining a new TimeSeries type, named 'MultiChannelTimeSeries'")
    },

    "schema": {
        "<MultiChannelTimeSeries>/": {
            "description": "Similar to ElectricalSeries, but without the restriction to volts",
            "merge": ["core:<TimeSeries>/"],
            "attributes": {
                "ancestry": {
                    "data_type": "text",
                    "dimensions": ["2"],
                    "value": ["TimeSeries", "MultiChannelTimeSeries"],
                    "const": True},
                "help": {
                    "data_type": "text",
                    "value": "A multi-channel time series",
                    "const": True}},
            "data": {
                "description": ("Multiple measurements are recorded at each point of time."),
                "dimensions": ["num_times", "num_channels"],
                "data_type": "float32"},
        }
    }
}}}


def parse_datetime(s):
    if s == 'unknown':
        val = None
    else:
        try:
            val = dateutil.parser.parse(s)
        except ValueError:
            val = s
    return val


class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file.
    """

    is_readable = True
    is_writable = True
    is_streameable = False
    #supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
    #                     SpikeTrain, Epoch, Event, ChannelIndex, Unit]
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal]
    readable_objects  = supported_objects
    writeable_objects = supported_objects

    has_header = False
    name = 'NWB'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'file'

    def __init__(self, filename) :
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename)
        self._extension_dir = tempfile.mkdtemp()
        extension_file = join(self._extension_dir, "nwb_neo_extension.py")
        with open(extension_file, "w") as fp:
            fp.write(str(neo_extension))
        self.extensions = [extension_file]

    def __del__(self):
        shutil.rmtree(self._extension_dir)

    def read_block(self, lazy=False, cascade=True, **kwargs):
        self._file = h5py.File(self.filename, 'r')
        self._lazy = lazy
        file_access_dates = self._file.get('file_create_date')
        if file_access_dates is None:
            file_creation_date = None
        else:
            file_access_dates = [parse_datetime(dt) for dt in file_access_dates]
            file_creation_date = file_access_dates[0]
        identifier = self._file.get('identifier').value
        if identifier == '_neo':  # this is an automatically generated name used if block.name is None
            identifier = None
        description = self._file.get('session_description').value
        if description == "no description":
            description = None
        block = Block(name=identifier,
                      description=description,
                      file_origin=self.filename,
                      file_datetime=file_creation_date,
                      rec_datetime=parse_datetime(self._file.get('session_start_time').value),
                      #index=?,
                      nwb_version=self._file.get('nwb_version').value,
                      file_access_dates=file_access_dates,
                      file_read_log='')
        if cascade:
            self._handle_general_group(block)
            self._handle_epochs_group(block)
            self._handle_acquisition_group(block)
            self._handle_stimulus_group(block)
            self._handle_processing_group(block)
            self._handle_analysis_group(block)
        self._lazy = False
        return block

    def write_block(self, block, **kwargs):

        # todo: handle block.file_datetime, block.file_origin, block.index, block.annotations

        if block.rec_datetime:
            start_time = block.rec_datetime.isoformat()
        else:
            start_time = "unknown"
        self._file = nwb_file.open(self.filename,
                                   start_time=start_time,
                                   mode="w",
                                   identifier=block.name or "_neo",
                                   description=block.description or "no description",
                                   core_spec="nwb_core.py", extensions=self.extensions,
                                   default_ns="core", keep_original=False, auto_compress=True)
        for segment in block.segments:
            self._write_segment(segment)
        self._file.close()

        if block.file_origin is None:
            block.file_origin = self.filename

        # The nwb module sets 'file_create_date' automatically. Here, if the block
        # already contains a 'file_datetime', we wish to set that as the first entry
        # in 'file_create_date'
        self._file = h5py.File(self.filename, "r+")
        nwb_create_date = self._file['file_create_date'].value
        if block.file_datetime:
            del self._file['file_create_date']
            self._file['file_create_date'] = np.array([block.file_datetime.isoformat(), nwb_create_date])
        else:
            block.file_datetime = parse_datetime(nwb_create_date[0])
        self._file.close()

    def _handle_general_group(self, block):
        block.annotations['file_read_log'] += ("general group not handled\n")

    def _handle_epochs_group(self, block):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        epochs = self._file.get('epochs')
        # todo: handle epochs.attrs.get('tags')
        for name, epoch in epochs.items():
            # todo: handle epoch.attrs.get('links')
            signals = []
            for key, value in epoch.items():
                if key == 'start_time':
                    t_start = value * pq.second
                elif key == 'stop_time':
                    t_stop = value * pq.second
                else:
                    # todo: handle value['count']
                    # todo: handle value['idx_start']
                    signals.append(self._handle_timeseries(key, value.get('timeseries')))
            segment = Segment(name=name)
            for signal in signals:
                signal.segment = segment
                if isinstance(signal, AnalogSignal):
                    segment.analogsignals.append(signal)
                elif isinstance(signal, IrregularlySampledSignal):
                    segment.irregularlysampledsignals.append(signal)
            segment.block = block
            block.segments.append(segment)

    def _handle_timeseries(self, name, timeseries):
        # todo: check timeseries.attrs.get('schema_id')
        # todo: handle timeseries.attrs.get('source')
        data_group = timeseries.get('data')
        if self._lazy:
            data = np.array(())
            lazy_shape = data_group.value.shape  # inefficient to load the data to get the shape
        else:
            data = data_group.value

        units = get_units(data_group)
        if 'starting_time' in timeseries:
            # AnalogSignal
            sampling_metadata = timeseries.get('starting_time')
            t_start = sampling_metadata.value * pq.s
            sampling_rate = sampling_metadata.attrs.get('rate') * pq.Hz
            assert sampling_metadata.attrs.get('unit') == 'Seconds'
            # todo: handle data.attrs['resolution']
            signal = AnalogSignal(data,
                                  units=units,
                                  sampling_rate=sampling_rate,
                                  t_start=t_start,
                                  name=name)
        elif 'timestamps' in timeseries:
            # IrregularlySampledSignal
            if self._lazy:
                time_data = np.array(())
            else:
                time_data = timeseries.get('timestamps')
                assert time_data.attrs.get('unit') == 'Seconds'
            signal = IrregularlySampledSignal(time_data.value,
                                              data,
                                              units=units,
                                              time_units=pq.second)
        else:
            raise Exception("Timeseries group does not contain sufficient time information")
        if self._lazy:
            signal.lazy_shape = lazy_shape
        return signal

    def _handle_acquisition_group(self, block):
        acq = self._file.get('acquisition')
        images = acq.get('images')
        if images and len(images) > 0:
            block.annotations['file_read_log'] += ("file contained {0} images; these are not currently handled by Neo\n".format(len(images)))

        # todo: check for signals that are not contained within an NWB Epoch,
        #       and create an anonymous Segment to contain them

    def _handle_stimulus_group(self, block):
        block.annotations['file_read_log'] += ("stimulus group not handled\n")

    def _handle_processing_group(self, block):
        block.annotations['file_read_log'] += ("processing group not handled\n")

    def _handle_analysis_group(self, block):
        block.annotations['file_read_log'] += ("analysis group not handled\n")

    def _write_segment(self, segment):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        epoch = nwb_utils.create_epoch(self._file, segment.name,
                                       time_in_seconds(segment.t_start),
                                       time_in_seconds(segment.t_stop))
        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            self._write_signal(signal, epoch, i)
        for spiketrain in segment.spiketrains:
            pass
        for event in segment.events:
            pass
        for neo_epoch in segment.epochs:
            pass

    def _write_signal(self, signal, epoch, i):
        # for now, we assume that all signals should go in /acquisition
        # this could be modified using Neo annotations
        signal_name = signal.name or "signal{0}".format(i)
        ts_name = "{0}_{1}".format(signal.segment.name, signal_name)
        ts = self._file.make_group("<MultiChannelTimeSeries>", ts_name, path="/acquisition/timeseries")
        conversion, base_unit = _decompose_unit(signal.units)
        attributes = {"conversion": conversion,
                      "unit": base_unit,
                      "resolution": float('nan')}
        if isinstance(signal, AnalogSignal):
            sampling_rate = signal.sampling_rate.rescale("Hz")
            # The following line is a temporary hack. IOs should not modify the objects
            # being written, but NWB only allows Hz, and the common IO tests
            # require units to be the same on write and read.
            # the proper solution is probably to add an option to the common IO tests
            # to allow different units
            signal.sampling_rate = sampling_rate
            ts.set_dataset("starting_time", time_in_seconds(signal.t_start),
                           attrs={"rate": float(sampling_rate)})
        elif isinstance(signal, IrregularlySampledSignal):
            ts.set_dataset("timestamps", signal.times.rescale('second').magnitude)
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
        ts.set_dataset("data", signal.magnitude,
                       dtype=np.float64,  #signal.dtype,
                       attrs=attributes)
        ts.set_dataset("num_samples", signal.shape[0])   # this is supposed to be created automatically, but is not
        #ts.set_dataset("num_channels", signal.shape[1])
        ts.set_attr("source", signal.name or "unknown")
        ts.set_attr("description", signal.description or "")

        nwb_utils.add_epoch_ts(epoch,
                               time_in_seconds(signal.segment.t_start),
                               time_in_seconds(signal.segment.t_stop),
                               signal_name,
                               ts)


def time_in_seconds(t):
    return float(t.rescale("second"))


def _decompose_unit(unit):
    """unit should be a Quantity object with unit magnitude

    Returns (conversion, base_unit_str)

    Example:

        >>> _decompose_unit(pq.nA)
        (1e-09, 'ampere')

    """
    assert isinstance(unit, pq.quantity.Quantity)
    assert unit.magnitude == 1
    conversion = 1.0
    def _decompose(unit):
        dim = unit.dimensionality
        if len(dim) != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt-metre
        uq, n = dim.items()[0]
        if n != 1:
            raise NotImplementedError("Compound units not yet supported")  # e.g. volt^2
        uq_def = uq.definition
        return float(uq_def.magnitude), uq_def
    conv, unit2 = _decompose(unit)
    while conv != 1:
        conversion *= conv
        unit = unit2
        conv, unit2 = _decompose(unit)
    return conversion, unit.dimensionality.keys()[0].name


prefix_map = {
    1e-3: 'milli',
    1e-6: 'micro',
    1e-9: 'nano'
}

def get_units(data_group):
    conversion = data_group.attrs.get('conversion')
    base_units = data_group.attrs.get('unit')
    return prefix_map[conversion] + base_units
